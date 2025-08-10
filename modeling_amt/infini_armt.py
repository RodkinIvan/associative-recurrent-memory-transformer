import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn.functional import relu as r
import torch.nn.functional as F
import wandb
from munch import Munch
import os

from modeling_amt.language_modeling import DPFP, AssociativeLayerWrapper

def split_tensor(tensor, segment_size, align='left'):
    if align in {'left', None}:
        split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
        segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
    elif align in {'right', None}:
        split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
        segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
    elif align == 'center':
        n_seg = math.ceil(tensor.shape[1] / segment_size)
        segments = torch.chunk(tensor, n_seg, dim=1)
    else:
        raise NotImplementedError
    return segments


class InfiniAssociativeLayerWrapper(AssociativeLayerWrapper):

    def __init__(self, layer, d_model,  num_mem_tokens, d_mem, n_heads=1, correction=True, info=None, use_denom=True, gating=False, **rmt_config) -> None:
        super().__init__(layer, d_model,  num_mem_tokens, d_mem, n_heads, correction, info, use_denom, gating)
        self.rmt_config = rmt_config
        self.tanh = torch.nn.Tanh()

    def full_association(self, hidden_states, model_output):
        self.zero_mem()

        output = []

        hidden_segments, model_segments = self.split_tensor(hidden_states), self.split_tensor(model_output)
        assert len(hidden_segments) == len(model_segments), f"{len(hidden_segments) != len(model_segments)}"

        for hidden, model_o in zip(hidden_segments, model_segments):
            assoc = self.associate(hidden)
            output.append(assoc) # it is intended that output doesn't include the model_output explicitely because it will be added in forward

            segment_output = assoc + model_o
            if not self.generate_mode:
                mem_tokens = segment_output[:, -self.num_mem_tokens:]
                self.update_mem(mem_tokens)
        out = torch.cat(output, dim=1)
        out = self.tanh(out)
        return out
        
    def forward(self, hidden_states, *args, **kwargs):
        out = list(self.layer(hidden_states, *args, **kwargs))
        association = self.full_association(hidden_states, out[0])
        out[0] = out[0] + association
        return tuple(out)
    
    def forward_no_update(self, hidden_states, *args, **kwargs):
        raise NotImplementedError

    def split_tensor(self, tensor):
        align = self.rmt_config.get('segment_alignment')
        segment_size = self.rmt_config.get('segment_size')
        return split_tensor(tensor, segment_size+self.num_mem_tokens, align)


class InfiniAssociativeMemoryCell(torch.nn.Module):
    def __init__(self, 
                 base_model, 
                 num_mem_tokens, 
                 d_mem, 
                 layers_attr: str = 'transformer.h', 
                 wrap_pos=False, 
                 correction=True, 
                 n_heads=1, 
                 use_denom=True, 
                 gating=False, 
                 freeze_mem=False,
                 act_on=False,
                 max_hop=4,
                 act_type='layer',
                 **rmt_config
        ):
        super().__init__()
        assert not act_on, "Not yet implemented"
        self.model = base_model
        
        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem
        self.d_model = base_model.get_input_embeddings().embedding_dim
        self.W_mem = []
        self.layers = self.model
        self.rmt_config = rmt_config

        self.layers_attrs = layers_attr.split('.')
        for i, attr in enumerate(self.layers_attrs):
            self.layers = getattr(self.layers, attr)
        
        for i in range(len(self.layers)):
            kw = dict(
                layer=self.layers[i], 
                d_model=self.d_model, 
                num_mem_tokens=self.num_mem_tokens, 
                d_mem=self.d_mem,
                correction=correction,
                info={'layer': i},
                n_heads=n_heads,
                use_denom=use_denom,
                gating=gating,
                **rmt_config
            )
            self.layers[i] = InfiniAssociativeLayerWrapper(**kw)

        self.create_memory(num_mem_tokens)
        self.wrap_pos = wrap_pos
        self.act_on = act_on
        if wrap_pos:
            self.wrap_positional_embeddings(num_mem_tokens)
        
        if freeze_mem:
            for layer in self.layers:
                layer.freeze_mem()
    
    def generate_mode(self, is_on):
        for layer in self.layers:
            layer.generate_mode = is_on
    
    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim =  getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim), device=embeddings.weight.data.device) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

    def wrap_positional_embeddings(self, num_mem_tokens):
        num_pos_embs, emb_dim = self.model.transformer.wpe.weight.shape
        prev_embs = self.model.transformer.wpe.weight.detach()
        self.model.transformer.wpe = torch.nn.Embedding(num_mem_tokens + num_pos_embs, emb_dim)

        new_num_pos = num_pos_embs + num_mem_tokens
        with torch.no_grad():
            self.model.transformer.wpe.weight[:len(self.model.transformer.wpe.weight)-num_mem_tokens] = prev_embs
        for layer in self.model.transformer.h:
            layer.layer.attn.bias = torch.tril(torch.ones((new_num_pos, new_num_pos), dtype=torch.uint8)).view(
                1, 1, new_num_pos, new_num_pos
            )

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def zero_mem(self):
        for layer in self.layers:
            layer.zero_mem()
            pass
    
    def detach_mem(self):
        for layer in self.layers:
            layer.detach_mem()
            pass

    def forward(self, input_ids, labels=None, labels_mask=None, zero_mem=False, attention_mask=None, **kwargs):
        if zero_mem:
            self.zero_mem()
        # print("="*50, "\n", input_ids.shape, "="*50)
        seg_kwargs = self.process_input(input_ids, **kwargs)
       
        out = self.model(attention_mask=attention_mask, **seg_kwargs)

        out = self.process_output(out, labels, labels_mask, **kwargs)

        return out, None

    def process_input(self, input_ids, **kwargs):
        seg_kwargs = dict(**kwargs)
        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        
        inputs_embeds = self.add_memory_tokens(inputs_embeds)
        
        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
            if kwargs.get('prev_attn_mask') is not None:
                seg_kwargs['attention_mask'] = torch.cat([kwargs['prev_attn_mask'], seg_kwargs['attention_mask']], dim=-1)
            if 'prev_attn_mask' in seg_kwargs:
                seg_kwargs.pop('prev_attn_mask')
        seg_kwargs['output_hidden_states'] = True

        # if self.wrap_pos:
        #     num_pos_embs = self.model.transformer.wpe.weight.shape[0]
        #     ordinary_pos = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        #     write_pos = torch.arange(num_pos_embs - self.num_mem_tokens, num_pos_embs, dtype=torch.long, device=input_ids.device)
        #     seg_kwargs['position_ids'] = torch.cat([
        #         ordinary_pos, 
        #         write_pos
        #     ]).long().unsqueeze(0)
        return seg_kwargs
    
    def pad_attention_mask(self, attention_mask):
        ones = torch.ones(attention_mask.shape[0], self.num_mem_tokens)
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            attention_segments = split_tensor(attention_mask, self.rmt_config['segment_size'], self.rmt_config.get('align'))
            mask = torch.cat([
                torch.cat([attn, ones], dim=1) for attn in attention_segments
            ], dim=1)
        
            return mask
    def remove_memory_tokens(self, tensor):
        segments = split_tensor(
                tensor, 
                self.rmt_config['segment_size'] + self.num_mem_tokens,
                self.rmt_config.get('align')
        )
        new_tensor = torch.cat([t[:, :-self.num_mem_tokens] for t in segments] , dim=1)
        return new_tensor
    
    def add_memory_tokens(self, tensor):
        memory_state = self.set_memory(tensor.shape)
        segments = split_tensor(tensor, self.rmt_config['segment_size'], self.rmt_config.get('align'))
        new_tensor = torch.cat([torch.cat([t, memory_state], dim=1) for t in segments], dim=1)
        return new_tensor
    
    def process_output(self, model_outputs, labels, labels_mask, **kwargs):
        if (self.num_mem_tokens not in {0, None}):
            out = CausalLMOutputWithCrossAttentions()
            out['logits'] = self.remove_memory_tokens(model_outputs.logits)
            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = self.remove_memory_tokens(model_outputs.hidden_states)
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            out = model_outputs

        if labels is not None:
            ce_loss_fn = CrossEntropyLoss()
            logits = out['logits'][..., :-1, :].contiguous()
            flat_logits = logits.view(-1, logits.size(-1))
            labels = labels[..., 1:].contiguous()
            flat_labels = labels.view(-1)
            if labels_mask is not None:
                flat_mask = labels_mask[..., :-1].contiguous().view(-1)
                flat_logits = flat_logits[flat_mask]
                flat_labels = flat_labels[flat_mask]
            ce_loss = ce_loss_fn(flat_logits, flat_labels)
            out['ce_loss'] = ce_loss

        if kwargs.get('use_cache', False):
            out['past_key_values'] = model_outputs.past_key_values
        if self.act_on and self.act_type == 'model':
            out['remainders'] = model_outputs['remainders']
            out['n_updates'] = model_outputs['n_updates']
        return out
    
    def generate(self, input_ids, attention_mask, zero_mem=False, **generate_kwargs):
        if zero_mem:
            self.zero_mem()
        
        
        self.generate_mode(True)
        seg_kwargs = self.process_input(input_ids, attention_mask=attention_mask)
        out = self.model.generate(
            inputs_embeds=seg_kwargs['inputs_embeds'][:, :-self.num_mem_tokens], 
            attention_mask=seg_kwargs['attention_mask'][:, :-self.num_mem_tokens], 
            **generate_kwargs
        )
        self.generate_mode(False)
        return out