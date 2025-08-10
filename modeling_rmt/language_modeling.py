import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, wrap_pos=False):
        super().__init__()
        self.model = base_model
        self.create_memory(num_mem_tokens)
        self.wrap_pos = wrap_pos
        if wrap_pos:
            self.wrap_positional_embeddings(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim =  getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def wrap_positional_embeddings(self, num_mem_tokens):

        num_pos_embs, emb_dim = self.model.transformer.wpe.weight.shape
        self.old_num_pos = num_pos_embs
        new_num_pos = num_pos_embs + 2 * num_mem_tokens
        prev_embs = self.model.transformer.wpe.weight.detach()
        self.model.transformer.wpe = torch.nn.Embedding(new_num_pos, emb_dim)
        with torch.no_grad():
            self.model.transformer.wpe.weight[num_mem_tokens:num_pos_embs+num_mem_tokens] = prev_embs
        for layer in self.model.transformer.h:
            layer.attn.bias = torch.tril(torch.ones((new_num_pos + num_pos_embs, new_num_pos + num_pos_embs), dtype=torch.uint8)).view(
                1, 1, new_num_pos + num_pos_embs, new_num_pos + num_pos_embs
            )

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, labels=None, labels_mask=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, **kwargs)
        out = self.model(**seg_kwargs)
        out, new_memory_state = self.process_output(out, labels, labels_mask, **kwargs)

        return out, new_memory_state
    
    def generate(self, input_ids, memory_state, attention_mask, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask)
        out = self.model.generate(
            inputs_embeds=seg_kwargs['inputs_embeds'][:, :-self.num_mem_tokens], 
            attention_mask=seg_kwargs['attention_mask'][:, :-self.num_mem_tokens], 
            **generate_kwargs
        )
        return out

    def process_input(self, input_ids, memory_state, **kwargs):
        seg_kwargs = dict(**kwargs)
        
        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)
        
        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
            if kwargs.get('prev_attn_mask') is not None:
                seg_kwargs['attention_mask'] = torch.cat([kwargs['prev_attn_mask'], seg_kwargs['attention_mask']], dim=-1)
            if 'prev_attn_mask' in seg_kwargs:
                seg_kwargs.pop('prev_attn_mask')
            
        seg_kwargs['output_hidden_states'] = True

        if self.wrap_pos:
            num_pos_embs = self.model.transformer.wpe.weight.shape[0]
            read_ordinary_pos = torch.arange(0, input_ids.size(1) + self.num_mem_tokens, dtype=torch.long, device=input_ids.device)
            write_pos = torch.arange(
                self.old_num_pos + self.num_mem_tokens, 
                self.old_num_pos + 2 * self.num_mem_tokens, 
                dtype=torch.long, 
                device=input_ids.device
            )
            seg_kwargs['position_ids'] = torch.cat([
                read_ordinary_pos, 
                write_pos
            ]).long().unsqueeze(0)

        return seg_kwargs
    
    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, self.num_mem_tokens:-self.num_mem_tokens] = attention_mask
            return mask
    
    def process_output(self, model_outputs, labels, labels_mask, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens:]
            out['logits'] = model_outputs.logits[:, self.num_mem_tokens:-self.num_mem_tokens]
            
            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, self.num_mem_tokens:-self.num_mem_tokens] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            memory_state = None
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
        if kwargs.get('use_cache') is not None:
            out['past_key_values'] = model_outputs.past_key_values
        return out, memory_state 


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
                input_segmented=False,
                ):
        sliding_window = self.rmt_config['sliding_window'] if 'sliding_window' in self.rmt_config else False
        memory_state = None
        if input_segmented:
            n_segs = input_ids.shape[1] if not (input_ids is None) else inputs_embeds.shape[1]

            segmented = [dict(
                input_ids=input_ids[:, i] if not (input_ids is None) else None, 
                inputs_embeds=inputs_embeds[:, i] if not (inputs_embeds is None) else None, 
                attention_mask=attention_mask[:, i],
                labels=labels[:, i] if not (labels is None) else None, 
                labels_mask=labels_mask[:, i] if not (labels_mask is None) else None, 
            ) for i in range(n_segs)]
            labels = torch.cat([labels[:, i] for i in range(n_segs)], dim=1)
            if labels_mask is not None:
                labels_mask = torch.cat([labels_mask[:, i] for i in range(n_segs)], dim=1)
        else:
            segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, labels_mask=labels_mask)
        cell_outputs = []
        past_key_values = None
        num_mem_tokens = self.memory_cell.num_mem_tokens
        prev_attn_mask = None
        for seg_num, segment in enumerate(segmented):
            seg_len = segment['input_ids'].size(-1)
            cell_out, memory_state = self.memory_cell(**segment, 
                                                      memory_state=memory_state, 
                                                      output_hidden_states=True, 
                                                      use_cache=sliding_window, 
                                                      past_key_values=past_key_values,
                                                      prev_attn_mask=prev_attn_mask
                                                    )
            
            if sliding_window:
                prev_attn_mask = segment['attention_mask'] * torch.triu(torch.ones_like(segment['attention_mask']))
                past_key_values = [
                    [
                        k_or_v[..., -(num_mem_tokens+seg_len):k_or_v.size(-2)-num_mem_tokens, :].detach() 
                        for k_or_v in seg_kv
                    ] 
                    for seg_kv in cell_out['past_key_values']
                ]
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

        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)

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
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])

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
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

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


class Distillator(torch.nn.Module):
    def __init__(self, teacher_model, student_model, alpha_distil):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha_distil
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    def forward(self, 
                input_ids, 
                labels=None, 
                labels_mask=None, 
                inputs_embeds=None, 
                attention_mask=None, 
                output_attentions=None, 
                output_hidden_states=None,
                input_segmented=False,
                sliding_window=False,
                ):
        with torch.no_grad():
            if self.training:
                if input_segmented:
                    n_segs = input_ids.shape[1] if not (input_ids is None) else inputs_embeds.shape[1]
                    teacher_output = self.teacher(
                        input_ids=torch.cat([input_ids[:, i] for i in range(n_segs)], dim=1) if input_ids is not None else None,
                        labels=torch.cat([labels[:, i] for i in range(n_segs)], dim=1)  if labels is not None else None, 
                        inputs_embeds=torch.cat([inputs_embeds[:, i] for i in range(n_segs)], dim=1) if inputs_embeds is not None else None, 
                        attention_mask=torch.cat([attention_mask[:, i] for i in range(n_segs)], dim=1) if attention_mask is not None else None, 
                        output_attentions=output_attentions, 
                        output_hidden_states=output_hidden_states
                    )
                else:
                    teacher_output = self.teacher(
                        input_ids=input_ids,
                        labels=labels, 
                        inputs_embeds=inputs_embeds, 
                        attention_mask=attention_mask, 
                        output_attentions=output_attentions, 
                        output_hidden_states=output_hidden_states
                    )
            else: 
                teacher_output = dict()
        student_output = self.student(
            input_ids=input_ids,
            labels=labels,
            labels_mask=labels_mask, 
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            input_segmented=input_segmented,
            sliding_window=sliding_window
        )
        
        if input_segmented:
            n_segs = input_ids.shape[1] if not (input_ids is None) else inputs_embeds.shape[1]
            labels = torch.cat([labels[:, i] for i in range(n_segs)], dim=1)
            if labels_mask is not None:
                labels_mask = torch.cat([labels_mask[:, i] for i in range(n_segs)], dim=1)
    
        out = self.process_outputs(teacher_output, student_output,
            labels=labels,
            labels_mask=labels_mask, 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states)

        return out
                  
    def process_outputs(self, teacher_output, student_output, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        teacher_logits = teacher_output.logits if self.training else None
        student_logits = student_output.logits

        for (k, v) in student_output.items():
            out[k] = v
            
            teachers = teacher_output.get(k)
            if teachers is not None:
                out[f'teacher_{k}'] = teachers

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_t_logits = teacher_logits[..., :-1, :].contiguous() if self.training else None

            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_t_logits = shift_t_logits.view(-1, shift_t_logits.size(-1)) if self.training else None
            
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]
                flat_t_logits = flat_t_logits[shift_mask.view(-1)] if self.training else None

            dist_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

            log_sftmx_student = torch.log_softmax(flat_logits, dim=-1)  
            log_sftmx_teacher = torch.log_softmax(flat_t_logits, dim=-1) if self.training else None
            dist = dist_fct(log_sftmx_student, log_sftmx_teacher) if self.training else None
            out['ce_loss'] = out['loss']
            if self.training:
                out['dist'] = dist
                out['loss'] = (1 - self.alpha) * out['ce_loss'] + self.alpha * dist
            
        else:
            out['loss'] = 0

        return out 

