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

    def forward(self, *args, state, inputs, fn, max_hop,  encoder_output=None, **kwargs):
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
                if isinstance(state, tuple) and len(state) > 1:
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
        
        
        
import torch
import torch.nn as nn
from transformers import MambaForCausalLM

import torch
import math


class AdaptiveMambaForCausalLM(MambaForCausalLM):
    def __init__(self, config):
        super(AdaptiveMambaForCausalLM, self).__init__(config)
        
        # Flag for enabling ACT, and max_hop
        self.use_act = config.use_act if hasattr(config, 'use_act') else False
        self.max_hop = config.max_hop if hasattr(config, 'max_hop') else 4
        
        # Initialize ACT layer if enabled
        if self.use_act:
            self.act_fn = ACT_basic(config.hidden_size)

    def forward(self, input_ids, labels=None, **kwargs):
        # re-init remindes and updates
        self.remainders = torch.zeros(1,)
        self.n_updates = torch.zeros(1,)
        self.segments_passed = torch.zeros(1,)
        
        x = self.backbone.embeddings(input_ids)
        self.remainders = self.remainders.to(x.device)
        self.n_updates = self.n_updates.to(x.device)
        self.segments_passed = self.segments_passed.to(x.device)
        if self.use_act:
            x, (remainders, n_updates) = self.act_fn(
                state=x,
                inputs=x,
                fn=lambda x: self.apply_layers(x),
                max_hop=self.max_hop,
            )
            self.remainders = self.remainders + remainders # 1 - \sum(h_i); L' = L + tau * mean(reminders)
            self.n_updates = self.n_updates + n_updates
            self.segments_passed = self.segments_passed + 1
        else:
            x = self.apply_layers(x)

        x = self.backbone.norm_f(x)
        logits = self.lm_head(x)
        return  {"logits": logits,
                    "n_updates": self.n_updates,
                    "remainders": self.remainders}

    def apply_layers(self, x):
        for layer in self.backbone.layers:
            x = layer(x)
        return x
        
