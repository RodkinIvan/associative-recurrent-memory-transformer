import sys
import os
os.environ["RWKV_TRAIN_TYPE"] = 'infctx'
os.environ["WKV"] = 'fla'
os.environ['RWKV_MY_TESTING'] = 'x060'
# os.environ["RWKV_FLOAT_MODE"] = "bf16"
import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from baselines.rwkv.RWKV_v5.src.model import RWKV as RWKV5
from baselines.rwkv.RWKV_v6.src.model import RWKV as RWKV6
from munch import Munch

class RWKVModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input_ids=None, inputs_embeds=None, state=None, attention_mask=None, *args, **kwargs):
        if state is None:
            state = (None, None)
        out, new_shift, new_wkv = self.model(idx=input_ids, embs=inputs_embeds, last_shift_states=state[0], last_wkv_states=state[1])
        return  Munch(
                    logits=out,
                    state=(new_shift, new_wkv)
                )
    
    def generate(self, input_ids, attention_mask, pad_token_id, max_new_tokens, state, max_length):
        generation_outputs = [[]]
        output = self.forward(input_ids=input_ids, state=state)
        out, state = output['logits'], output['state']
        device = next(self.model.parameters()).device
        assert input_ids.size(0) == 1
        for i in range(max_new_tokens):
            token = out[0, -1].argmax(-1)
            generation_outputs[0].append(token)
            out, state = self.forward(
                input_ids=torch.tensor([[token.item()]],dtype=torch.long, device=device), 
                state=state)
        return generation_outputs

    def get_input_embeddings(self):
        return self.model.emb
        
class RWKV_v5_tiny(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.model = RWKV5(**args)

    @staticmethod
    def from_pretrained(*_args, **kwargs):
        args = dict(
            load_model='/home/rodkin/lab/rwkv_tiny.pth',
            grad_cp=False
        )
        model = RWKV_v5_tiny(**args)
        
        return model

    def forward(self, input_ids, state=None, attention_mask=None):
        if state is None:
            state = (None, None)
        assert state[0] is None or isinstance(state[0], torch.Tensor)
        assert state[1] is None or isinstance(state[1], torch.Tensor)
        out, new_shift, new_wkv = self.model(idx=input_ids, last_shift_states=state[0], last_wkv_states=state[1])
        return out, (new_shift, new_wkv)
    
    def generate(self, input_ids, attention_mask, pad_token_id, max_new_tokens, state, max_length=None):
        generation_outputs = torch.zeros(len(input_ids), max_new_tokens)
        out, state = self.forward(input_ids=input_ids, state=state)
        device = next(self.model.parameters()).device
        for i in range(max_new_tokens):
            token = out[:, -1].argmax(-1)
            generation_outputs[:, i] = token
            out, state = self.forward(
                input_ids=torch.tensor(token[..., None],dtype=torch.long, device=device), 
                state=state)
        return generation_outputs
class RWKV_v6(RWKVModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = RWKV6(*args, **kwargs)
        self.config = Munch(
            n_embd=self.model.n_embd,
            hidden_size=self.model.n_embd
        )
    @staticmethod
    def from_pretrained(*_args, **kwargs):
        load = _args[0]
        args = dict(
            load_model=load,
            grad_cp=True
        )
        model = RWKV_v6(**args)
        return model

class RWKV_v5(RWKVModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = RWKV5(*args, **kwargs)
        self.config = Munch(
            n_embd=self.model.n_embd,
            hidden_size=self.model.n_embd
        )
    @staticmethod
    def from_pretrained(*_args, **kwargs):
        load = _args[0]
        args = dict(
            load_model=load,
            grad_cp=True
        )
        model = RWKV_v5(**args)
        return model

class MemoryCell(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, input_ids, memory_state=None, labels=None, labels_mask=None, **kwargs):
        seg_kwargs = self.process_input(input_ids, memory_state, **kwargs)
        output = self.model(**seg_kwargs)
        out, state = output['logits'], output['state']
        out = self.process_output(out, labels, labels_mask, **kwargs)

        return out, state
    
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
        seg_kwargs['state'] = memory_state
        del seg_kwargs['attention_mask']

        return seg_kwargs
    
    def process_output(self, model_outputs, labels, labels_mask, **kwargs):
        out = Munch()
        out.logits = model_outputs
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
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs
    def forward(self, 
                input_ids, 
                labels=None, 
                labels_mask=None, 
                inputs_embeds=None, 
                attention_mask=None, 
                output_attentions=None, 
                output_hidden_states=None,
                input_segmented=None,
                sliding_window=None,
                output_only_last_segment=False,
                ):
        memory_state = None

        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, labels_mask=labels_mask)
        cell_outputs = []
        for seg_num, segment in enumerate(segmented):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state)
            
            if (not output_only_last_segment) or (seg_num == len(segmented) - 1):
                cell_outputs.append(cell_out)
            
            self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states)
        return out

        
    def generate(self, input_ids, attention_mask, **generate_kwargs):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        for _, segment in enumerate(segmented[:-1]):
            _, memory_state = self.memory_cell(**segment, memory_state=memory_state)

        final_segment = segmented[-1]
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

        return out

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments
    
    def split_tensor(self, tensor):
        align = self.rmt_config.get('segment_alignment')
        segment_size = self.rmt_config.get('segment_size')
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

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        labels = kwargs.get('labels')
        labels = labels[:, -full_logits.size(1):]
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            labels_mask = labels_mask[:, -full_logits.size(1):]
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)] #
                flat_logits = flat_logits[shift_mask.view(-1)]
                
            out['loss'] = loss_fct(flat_logits, flat_labels)
        else:
            out['loss'] = 0

        out['ce_loss'] = out['loss']
        
        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f'{key}_{seg_num}'] = value

        return out 
        
    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return True
        
        memory_state = memory_state.detach()
        return False