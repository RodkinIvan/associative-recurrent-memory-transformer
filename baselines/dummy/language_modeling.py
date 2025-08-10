import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from munch import Munch

class MemoryCell(torch.nn.Module):
    def __init__(self, base_model,
                 ):
        super().__init__()
        self.model = base_model
        
        
    def forward(self, input_ids, memory_state=None, **kwargs):
        out = self.model(input_ids)
        return out, memory_state
    
    def generate(self, input_ids, memory_state, attention_mask, **generate_kwargs):
        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask)
        out = self.model.generate(
            **seg_kwargs,
            **generate_kwargs
        )
        return out

    def process_input(self, input_ids, memory_state, **kwargs):
        seg_kwargs = dict(**kwargs)
        
        seg_kwargs['input_ids'] = input_ids

        return seg_kwargs
    
    def process_output(self, model_outputs, labels, labels_mask, **kwargs):
        out = Munch()
        out.logits = model_outputs.logits

        if labels is not None:
            ce_loss_fn = CrossEntropyLoss()
            logits = out.logits[..., :-1, :].contiguous()
            flat_logits = logits.view(-1, logits.size(-1))
            labels = labels[..., 1:].contiguous()
            flat_labels = labels.view(-1)
            if labels_mask is not None:
                flat_mask = labels_mask[..., :-1].contiguous().view(-1)

                flat_logits = flat_logits[flat_mask]
                flat_labels = flat_labels[flat_mask]
            ce_loss = ce_loss_fn(flat_logits, flat_labels)
            out.ce_loss = ce_loss
        
        return out

class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, 
                 time_penalty=0,
                 act_on=False,
                 **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs
        self.time_penalty = time_penalty
        self.act_on = act_on
        
        
    def forward(self, 
                input_ids, 
                labels=None, 
                labels_mask=None, 
                inputs_embeds=None, 
                attention_mask=None, 
                output_attentions=None, 
                output_hidden_states=None,
                ):
        memory_state = None
        segment = dict(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, labels_mask=labels_mask)
        cell_outputs = []
        cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state)
        cell_outputs.append(cell_out)

        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states)
        return out

        
    def generate(self, input_ids, attention_mask, **generate_kwargs):
        memory_state = None
        segmented = dict(input_ids=input_ids, attention_mask=attention_mask)

        for segment in segmented[:-1]:
            _, memory_state = self.memory_cell(**segment, memory_state=memory_state)

        segment = segmented[-1]
        out = self.memory_cell.generate(**segment, memory_state=memory_state, **generate_kwargs)
        return out

    def split(self, tensor, first_seg_len):
        if first_seg_len is None or tensor.size(1) <= first_seg_len:
            return [tensor]
        else:
            return [tensor[:, :first_seg_len],] + [tensor[:, i:i+1] for i in range(first_seg_len, tensor.size(1))]
        
    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o["logits"] for o in cell_outputs], dim=1)
        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]
                
            out['loss'] = loss_fct(flat_logits, flat_labels)
        else:
            out['loss'] = 0

        out['ce_loss'] = out['loss']
        
        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if self.act_on:
            remainders = []
            n_updates = []
            for o in cell_outputs:
                    remainders.extend(o['remainders'])
                    n_updates.extend(o['n_updates'])
            remainders = torch.mean(torch.stack(remainders, dim=0))
            n_updates = torch.mean(torch.stack(n_updates, dim=0))
            out['n_updates'] = n_updates.detach().cpu()
            out['remainders'] = remainders.detach().cpu()
            out['loss'] = out['loss'] + self.time_penalty * remainders
        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f'{key}_{seg_num}'] = value

        return out 
    
