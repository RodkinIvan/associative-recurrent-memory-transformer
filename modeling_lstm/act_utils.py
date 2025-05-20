from torch import nn
import torch
import numpy as np
import math


def gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = ( math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 
                    'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)

class ACT_basic(nn.Module):
    def __init__(self,hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)  
        self.p.bias.data.fill_(1) 
        self.threshold = 1 - 0.1

    def forward(self, *args, state, inputs, fn, time_enc, pos_enc, max_hop,  encoder_output=None, **kwargs):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        remainders = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        # for l in range(self.num_layers):
        rest = None
        while( ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            # state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            # state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if(encoder_output):
                state, _ = fn((state,encoder_output))
            else:
                # apply transformation on the state
                state = fn(state, *args, **kwargs)
                if isinstance(state, tuple):
                    rest = state[1:]
                    state = state[0]

            # update running part in the weighted state and keep the rest
            # print(state.shape, previous_state.shape, update_weights.shape)
            # print(state.dtype, previous_state.dtype, update_weights.dtype)
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop 
            ## to save a line I assigned to previous_state so in the next 
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1
        if rest is None:
            return previous_state, (remainders,n_updates)
        else:
            return (previous_state, *rest), (remainders, n_updates)


class ACT_constant_depth():
    def __init__(self):
        super(ACT_constant_depth, self).__init__()

    def __call__(self, *args, state, inputs, fn, time_enc, pos_enc, max_hop,  encoder_output=None, **kwargs):
        print("CONSTANT DEPTH TRUE")
        # init_hdd
        ## [B, S]
        remainders = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        # for l in range(self.num_layers):
        rest = None
        while(step < max_hop):
            # Add timing signal
            # state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            # state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)

            print('step:', step)

            if(encoder_output):
                state, _ = fn((state,encoder_output))
            else:
                # apply transformation on the state
                state = fn(state, *args, **kwargs)
                if isinstance(state, tuple):
                    rest = state[1:]
                    state = state[0]

            print("STATE SHAPE", state.shape)
            
            # update running part in the weighted state and keep the rest
            # print(state.shape, previous_state.shape, update_weights.shape)
            # print(state.dtype, previous_state.dtype, update_weights.dtype)
            previous_state = state
            ## previous_state is actually the new_state at end of hte loop 
            ## to save a line I assigned to previous_state so in the next 
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1
        if rest is None:
            return previous_state, (remainders,n_updates)
        else:
            return (previous_state, *rest), (remainders, n_updates)


class AdaptiveLayerWrapper(nn.Module):
    def __init__(self, 
                 layer, 
                 d_model, 
                 max_hop, 
                 constant_depth=False,                
                ) -> None:
        super().__init__()
        self.act = ACT_basic(d_model,)
        self.depth = max_hop
        self.max_length = 1024
        self.layer = layer
        self.constant_depth = constant_depth

        self.timing_signal = gen_timing_signal(self.max_length, d_model)
        ## for t
        self.position_signal = gen_timing_signal(self.depth, d_model)



    def forward(self, hidden_states, *args, **kwargs):
        # self.remainders = torch.zeros(1,)
        # self.n_updates = torch.zeros(1,)
        # self.remainders = self.remainders.to(hidden_states.device)
        # self.n_updates = self.n_updates.to(hidden_states.device)

        fwd = self.layer
        out, (remainders, n_updates) = self.act(
            *args,
            state=hidden_states, 
            inputs=hidden_states, 
            fn=fwd,
            time_enc=self.timing_signal,
            pos_enc=self.position_signal,
            max_hop=self.depth,
            constant_depth=self.constant_depth,
            **kwargs
        )

        # self.remainders = self.remainders + remainders # 1 - \sum(h_i); L' = L + tau * mean(reminders)
        # self.n_updates = self.n_updates + n_updates
        
        return out, (remainders, n_updates)

