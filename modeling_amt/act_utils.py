from torch import nn
import torch
import numpy as np
import math

from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
        self.eps = 0.1

    def forward(self, *args, state, inputs, fn, time_enc, pos_enc, max_hop,  encoder_output=None, **kwargs):
        # init_hdd
        ## [B, S]
        noisy_halting = False
        if 'noisy_halting' in kwargs:
            noisy_halting = kwargs['noisy_halting']
            kwargs.pop('noisy_halting')
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
            if noisy_halting and self.training:
                p = p + torch.randn_like(p) * self.eps
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


class ACTForWholeARMT(nn.Module):
    def __init__(self,hidden_size):
        super(ACTForWholeARMT, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)  
        self.p.bias.data.fill_(1) 
        self.threshold = 1 - 0.1

    def forward(self, *args, state, inputs, fn_no_update, fn_update, time_enc, pos_enc, max_hop,  encoder_output=None, **kwargs):
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
        while( ((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any()):
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
                if ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any():
                    state, _ = fn_no_update((state,encoder_output))
                else:
                    state, _ = fn_update((state, encoder_output))
            else:
                # apply transformation on the state
                if ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any():
                    state = fn_no_update(state, *args, **kwargs)
                else:
                    state = fn_update(state, *args, **kwargs)
                if isinstance(state, tuple):
                    rest = state[1:]
                    state = state[0]

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop 
            ## to save a line I assigned to previous_state so in the next 
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1
        if rest is None:
            return previous_state, (remainders,n_updates)
        else:
            return (previous_state, *rest), (remainders, n_updates)


class ACT_transformer(nn.Module):
    def __init__(self, hidden_size, num_heads=4, num_transformer_layers=1, dropout=0.1):
        super(ACT_transformer, self).__init__()
        # Transformer encoder
        transformer_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            norm_first=True
        )
        self.transformer = TransformerEncoder(transformer_layer, 
                                              num_layers=num_transformer_layers)
        
        # Feedforward layer for logits
        self.logit_ff = nn.Linear(hidden_size, 1)  
        self.logit_ff.bias.data.fill_(1)
        
        # Halting threshold
        self.sigma = nn.Sigmoid()
        self.threshold = 1 - 0.1

    def generate_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, *args, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None, **kwargs):
        batch_size, seq_len, hidden_size = inputs.shape
        halting_probability = torch.zeros(batch_size, seq_len).cuda()
        remainders = torch.zeros(batch_size, seq_len).cuda()
        n_updates = torch.zeros(batch_size, seq_len).cuda()
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        rest = None

        causal_mask = self.generate_causal_mask(seq_len).cuda()

        while ((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any():
            state_transformed = self.transformer(
                state.permute(1, 0, 2),  # [S, B, H]
                mask=causal_mask
            )  # [S, B, H]
            state_transformed = state_transformed.permute(1, 0, 2)  # [B, S, H]

            # Pass through linear layer and sigmoid
            p = self.sigma(self.logit_ff(state_transformed)).squeeze(-1)  # [B, S]

            # Update halting logic
            still_running = (halting_probability < 1.0).float()
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running
            halting_probability = halting_probability + p * still_running
            remainders = remainders + new_halted * (1 - halting_probability)
            halting_probability = halting_probability + new_halted * remainders
            n_updates = n_updates + still_running + new_halted
            update_weights = p * still_running + new_halted * remainders

            if encoder_output is not None:
                state, _ = fn((state, encoder_output))
            else:
                state = fn(state, *args, **kwargs)
                if isinstance(state, tuple):
                    rest = state[1:]
                    state = state[0]

            previous_state = (
                (state * update_weights.unsqueeze(-1)) +
                (previous_state * (1 - update_weights.unsqueeze(-1)))
            )
            step += 1

        if rest is None:
            return previous_state, (remainders, n_updates)
        else:
            return (previous_state, *rest), (remainders, n_updates)
