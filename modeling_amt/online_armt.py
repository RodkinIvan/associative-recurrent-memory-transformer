import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.cache_utils import Cache, DynamicCache
from torch.nn.functional import relu as r
import torch.nn.functional as F
import wandb
from munch import Munch
import os
from fla.ops.delta_rule.chunk import chunk_delta_rule

from modeling_amt.act_utils import ACT_basic, gen_timing_signal
# from baselines.rwkv.language_modeling import RWKVModel

def dpfp(x, nu=1):
  x = torch.cat([r(x), r(-x)], dim=-1)
  x_rolled = torch.cat([x.roll(shifts=j, dims=-1)
           for j in range(1,nu+1)], dim=-1)
  x_repeat = torch.cat([x] * nu, dim=-1)
  return x_repeat * x_rolled

class DPFP:
    def __init__(self, nu):
        self.nu = nu
    
    def __call__(self, x):
        nu = self.nu
        x = torch.cat([r(x), r(-x)], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1,nu+1)], dim=-1)
        x_repeat = torch.cat([x] * nu, dim=-1)
        return x_repeat * x_rolled
def attn_mask_to_4d(attn_mask, upper, query_len):
    if attn_mask is None:
        return None
    seg_len = attn_mask.size(-1)
    if upper:
        tri = torch.triu(torch.ones(query_len, seg_len))
    else:
        tri = torch.tril(torch.ones(query_len, seg_len))

    mask = torch.einsum('bj,ij->bij', attn_mask, tri.to(attn_mask.device))
    mask = mask.unsqueeze(1)
    return mask

def invert_attn_mask(attn_mask, dtype):
        min_dtype = torch.finfo(dtype).min
        new_mask = (1.0 - attn_mask) * min_dtype
        return new_mask



class AssociativeLayerWrapper(torch.nn.Module):

    def __init__(self, layer, d_model,  num_mem_tokens, d_mem, correction=True, info=None, use_denom=True, gating=False, num_heads=8) -> None:
        super().__init__()
        self.info = info
        self.seg_num = 0
        self.d_model = d_model
        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem
        self.num_heads = num_heads
        
        nu = 3
        self.d_key = 2 * nu * d_mem
        
        # Calculate head dimension to ensure it's <= 256
        self.head_dim = min(256, self.d_key // num_heads)
        # Adjust num_heads if needed to ensure d_key is divisible
        assert self.d_key % self.head_dim == 0, f"d_key {self.d_key} is not divisible by head_dim {self.head_dim}"
        self.num_heads = self.d_key // self.head_dim
        # self.d_key = self.num_heads * self.head_dim  # Ensure divisibility

        self.phi = DPFP(nu)
        # self.d_key = d_mem
        # self.phi = torch.nn.Identity()

        self.use_denom = use_denom

        self.W_mq = torch.nn.Linear(d_model, d_mem, bias=False)
        self.W_mk = torch.nn.Linear(d_model, d_mem, bias=False)
        self.W_mv = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_mb = torch.nn.Linear(d_model, self.num_heads)
        torch.nn.init.zeros_(self.W_mv.weight)
        


        # W_mem has shape (1, num_heads, head_dim, d_model // num_heads)
        head_value_dim = d_model // self.num_heads
        self.W_mem = torch.zeros(1, self.num_heads, self.head_dim, head_value_dim, dtype=torch.bfloat16)
        self.W_mem.requires_grad_(False)
        
        # self.ln = torch.nn.LayerNorm(d_model)

        self.zero_mem()
    
        self.layer = layer
        
        self.generate_mode = False
        self.correction = correction


    def associate(self, hidden_states):
        bsz, seq_len, d_model = hidden_states.shape

        # Ensure hidden_states is in bfloat16
        hidden_states = hidden_states.to(torch.bfloat16)
        
        self.W_mem = self.W_mem.to(hidden_states.device).to(torch.bfloat16)

        q = self.W_mq(hidden_states)

        mq = self.phi(q) # (bsz, seq_len, d_key)
        mq = F.normalize(mq, dim=-1, p=2.0)
        
        # Reshape mq to multiple heads: (bsz, seq_len, num_heads, head_dim)
        mq = mq.view(bsz, seq_len, self.num_heads, self.head_dim)

        # online part
        k = self.W_mk(hidden_states)
        hk = self.phi(k)
        hk = F.normalize(hk, dim=-1, p=2.0)
        # Reshape hk to multiple heads: (bsz, seq_len, num_heads, head_dim)
        hk = hk.view(bsz, seq_len, self.num_heads, self.head_dim)

        hv = self.W_mv(hidden_states)
        # Split values into heads: (bsz, seq_len, num_heads, d_model // num_heads)
        head_value_dim = d_model // self.num_heads
        hv = hv.view(bsz, seq_len, self.num_heads, head_value_dim)

        hb = torch.sigmoid(self.W_mb(hidden_states))

        # W_mem has shape: (1, num_heads, head_dim, d_model // num_heads)
        
        # Compute num with multi-head einsum and concatenate results
        num_heads = torch.einsum('bqhd,bhdk->bqhk', mq, self.W_mem)  # (bsz, seq_len, num_heads, d_model // num_heads)
        num = num_heads.reshape(bsz, seq_len, d_model)  # Concatenate heads back to full dimension

        # Ensure all tensors are explicitly in bfloat16
        mq = mq.to(torch.bfloat16)
        hk = hk.to(torch.bfloat16)
        hv = hv.to(torch.bfloat16)
        hb = hb.to(torch.bfloat16)
        
        local_associations, _ = chunk_delta_rule(
            mq,  # (bsz, seq_len, num_heads, head_dim)
            hk,  # (bsz, seq_len, num_heads, head_dim)
            hv,  # (bsz, seq_len, num_heads, d_model // num_heads)
            hb,  # (bsz, seq_len, num_heads)
            initial_state=self.W_mem,  # (1, num_heads, head_dim, d_model // num_heads)
            output_final_state=False
        )

        # local_associations has shape [B, T, H, V], concatenate heads to get [B, T, V]
        local_associations = local_associations.reshape(bsz, seq_len, d_model)  # Concatenate heads back to full dimension

        hidden_states = num + local_associations

        return hidden_states


    
    def construct_associations(self, k, v, b):
        # Reshape inputs for multi-head computation
        bsz, num_mem_tokens, _ = k.shape
        k_heads = k.view(bsz, num_mem_tokens, self.num_heads, self.head_dim)
        v_heads = v.view(bsz, num_mem_tokens, self.num_heads, self.d_model // self.num_heads)
        
        # For non-gating, b has shape (bsz, num_mem_tokens, num_heads)
        b_heads = b
        einop = "bqhd,bqhv,bqh->bhdv"
        
        associations = torch.einsum(einop, k_heads, v_heads, b_heads)  # (bsz, num_heads, head_dim, d_model // num_heads)
        return associations
    
    def forward(self, hidden_states, *args, **kwargs):
        hidden_states = self.associate(
                hidden_states
        ) + hidden_states
        out = self.layer(hidden_states, *args, **kwargs)
        if not self.generate_mode:
            mem_tokens = out[0][:, -self.num_mem_tokens:]
            self.update_mem(mem_tokens)
        return out
    
    def forward_no_update(self, hidden_states, *args, **kwargs):
        hidden_states = self.associate(
            # self.ln(
                hidden_states
            # )
        ) + hidden_states
        out = self.layer(hidden_states, *args, **kwargs)
        return out

    def update_mem(self, mem_tokens):
        # Ensure mem_tokens is in bfloat16
        mem_tokens = mem_tokens.to(torch.bfloat16)

        self.W_mem = self.W_mem.to(mem_tokens.device).to(torch.bfloat16)
        k = self.W_mk(mem_tokens)
        mk = self.phi(k)
        mk = F.normalize(mk, dim=-1, p=2.0).to(mem_tokens.device).to(torch.bfloat16)
        mk_heads = mk.view(mk.size(0), mk.size(1), self.num_heads, self.head_dim)

        mq_heads = torch.zeros_like(mk_heads).to(mem_tokens.device).to(torch.bfloat16)

        new_mv = self.W_mv(mem_tokens) # (bsz, num_mem_tokens, d_model)
        mv_heads = new_mv.view(new_mv.size(0), new_mv.size(1), self.num_heads, self.d_model // self.num_heads)

        mb_heads = torch.sigmoid(self.W_mb(mem_tokens))
        _, final_state = chunk_delta_rule(
            mq_heads,
            mk_heads,
            mv_heads,
            mb_heads,
            initial_state=self.W_mem,
            output_final_state=True
        )

        self.W_mem = final_state
        self.seg_num += 1

    def freeze_mem(self):
        self.W_mb.weight.requires_grad = False
        self.W_mb.bias.requires_grad = False

        self.W_mq.weight.requires_grad = False
        self.W_mk.weight.requires_grad = False
        self.W_mv.weight.requires_grad = False

    def zero_mem(self):
        # Use bfloat16 to match the mixed precision training
        # W_mem has shape: (1, num_heads, head_dim, d_model // num_heads)
        head_value_dim = self.d_model // self.num_heads
        self.W_mem = torch.zeros(1, self.num_heads, self.head_dim, head_value_dim, dtype=torch.bfloat16)
        self.seg_num = 0



class AdaptiveAssociativeLayerWrapper(AssociativeLayerWrapper):
    def __init__(self, 
                 layer, 
                 d_model, 
                 num_mem_tokens, 
                 d_mem, 
                 max_hop,
                 correction=True, 
                 info=None, 
                 use_denom=True, 
                 gating=False,
                 num_heads=8
                ) -> None:
        super().__init__(layer, d_model, num_mem_tokens, d_mem, correction, info, use_denom, gating, num_heads)
        self.act = ACT_basic(d_model)
        self.depth = max_hop
        self.max_length = 1024

        self.timing_signal = gen_timing_signal(self.max_length, d_model)
        ## for t
        self.position_signal = gen_timing_signal(self.depth, d_model)

        self.remainders = torch.zeros(1,)
        self.n_updates = torch.zeros(1,)
        self.segments_passed = torch.zeros(1,)

    def associate(self, hidden_states):
        self.remainders = self.remainders.to(hidden_states.device)
        self.n_updates = self.n_updates.to(hidden_states.device)
        self.segments_passed = self.segments_passed.to(hidden_states.device)
        out, (remainders, n_updates) = self.act(
            state=hidden_states, 
            inputs=hidden_states, 
            fn=super().associate,
            time_enc=self.timing_signal,
            pos_enc=self.position_signal,
            max_hop=self.depth
        )
        
        self.remainders = self.remainders + remainders # 1 - \sum(h_i); L' = L + tau * mean(remainders)
        self.n_updates = self.n_updates + n_updates
        self.segments_passed = self.segments_passed + 1
        return out
    
    def zero_mem(self):
        self.remainders = torch.zeros(1,)
        self.n_updates = torch.zeros(1,)
        self.segments_passed = torch.zeros(1,)
        return super().zero_mem()
    



class AdaptiveAssociativeLayerWrapper2(AssociativeLayerWrapper):
    def __init__(self, 
                 layer, 
                 d_model, 
                 num_mem_tokens, 
                 d_mem, 
                 max_hop,
                 correction=True, 
                 info=None, 
                 use_denom=True, 
                 gating=False,
                 num_heads=8
                ) -> None:
        super().__init__(layer, d_model, num_mem_tokens, d_mem, correction, info, use_denom, gating, num_heads)
        self.act = ACT_basic(d_model)
        self.depth = max_hop
        self.max_length = 1024

        self.timing_signal = gen_timing_signal(self.max_length, d_model)
        ## for t
        self.position_signal = gen_timing_signal(self.depth, d_model)

        self.remainders = torch.zeros(1,)
        self.n_updates = torch.zeros(1,)
        self.segments_passed = torch.zeros(1,)

    def forward(self, hidden_states, *args, **kwargs):
        self.remainders = self.remainders.to(hidden_states.device)
        self.n_updates = self.n_updates.to(hidden_states.device)
        self.segments_passed = self.segments_passed.to(hidden_states.device)

        fwd = super().forward_no_update
        out, (remainders, n_updates) = self.act(
            *args,
            state=hidden_states, 
            inputs=hidden_states, 
            fn=fwd,
            time_enc=self.timing_signal,
            pos_enc=self.position_signal,
            max_hop=self.depth,
            **kwargs
        )
        if not self.generate_mode:
            mem_tokens = out[0][:, -self.num_mem_tokens:]
            # mem_tokens = out[0]
            self.update_mem(mem_tokens)
        self.remainders = self.remainders + remainders # 1 - \sum(h_i); L' = L + tau * mean(reminders)
        self.n_updates = self.n_updates + n_updates
        self.segments_passed = self.segments_passed + 1
        return out

    
    def zero_mem(self):
        self.remainders = torch.zeros(1,)
        self.n_updates = torch.zeros(1,)
        self.segments_passed = torch.zeros(1,)
        return super().zero_mem()


class AssociativeMemoryCell(torch.nn.Module):
    def __init__(self, 
                 base_model, 
                 num_mem_tokens, 
                 d_mem, 
                 layers_attr: str = 'model.layers', 
                 wrap_pos=False, 
                 correction=True, 
                 use_denom=True, 
                 gating=False, 
                 freeze_mem=False,
                 act_on=False,
                 max_hop=4,
                 act_type='associative',
                 attend_to_previous_input=False,
                 use_sink=False,
                 num_heads=8
        ):
        super().__init__()
        self.model = base_model
        self.attend_to_previous_input = attend_to_previous_input
        self.previous_input = None
        self.use_sink = use_sink

        self.RWKV_ARMT = False #isinstance(self.model, RWKVModel)

        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem
        self.d_model = base_model.get_input_embeddings().embedding_dim
        self.W_mem = []
        self.layers = self.model

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
                use_denom=use_denom,
                gating=gating,
                num_heads=num_heads
            )
            if act_on:
                kw['max_hop'] = max_hop
            if not act_on:
                self.layers[i] = AssociativeLayerWrapper(**kw)
            elif act_type == 'associative':
                self.layers[i] = AdaptiveAssociativeLayerWrapper(**kw)
            elif act_type == 'layer':
                self.layers[i] = AdaptiveAssociativeLayerWrapper2(**kw)
            else:
                raise f'Unknown ACT type: {act_type}'
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
        if self.use_sink:
            self.sink = torch.nn.Parameter(torch.randn((1, memory_dim), device=embeddings.weight.data.device), requires_grad=True)


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
        if self.use_sink:
            sink = self.sink.repeat(input_shape[0], 1, 1)
        else:
            sink = None
        return memory, sink

    def zero_mem(self):
        for layer in self.layers:
            layer.zero_mem()
            pass
        self.previous_input = None

    def forward(self, input_ids, labels=None, labels_mask=None, zero_mem=False, **kwargs):
        current_input_ids = input_ids.clone()
        if self.attend_to_previous_input and self.previous_input is not None:
            input_ids = torch.cat([self.previous_input, input_ids], dim=1)
        
        if zero_mem:
            self.zero_mem()
        seg_kwargs = self.process_input(input_ids, **kwargs)

        if self.RWKV_ARMT and not self.layers[0].generate_mode:
            input1 = dict()
            input2 = dict()
            for item in seg_kwargs:
                if isinstance(seg_kwargs[item], torch.Tensor):
                # if False:
                    input1[item] = seg_kwargs[item][:, :-self.num_mem_tokens]
                    input2[item] = seg_kwargs[item][:, -self.num_mem_tokens:]
                else:
                    input1[item] = seg_kwargs[item]
                    input2[item] = seg_kwargs[item]
            
            self.generate_mode(True)
            out = self.model(**input1)
            self.generate_mode(False)
            state_tmp = tuple([torch.clone(state) for state in out['state']])
            out = Munch({k: torch.clone(t) if isinstance(t, torch.Tensor) else t for k, t in out.items()})
            input2['state'] = out['state']
            _ = self.model(**input2)
            out['state'] = state_tmp
            # out['state'] = out2['state']
            # out = self.model(**seg_kwargs)
            # out['logits'] = out['logits'][:, :-self.num_mem_tokens]
        else:
            out = self.model(**seg_kwargs)

        if self.attend_to_previous_input and self.previous_input is not None:
            out['logits'] = out['logits'][:, self.previous_input.size(1):]
        out = self.process_output(out, labels, labels_mask, **kwargs)
        self.previous_input = current_input_ids
        return out

    def process_input(self, input_ids, **kwargs):
        memory_state, sink = self.set_memory(input_ids.shape)
        seg_kwargs = dict(**kwargs)
        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if self.use_sink:
            inputs_embeds = torch.cat([sink, inputs_embeds, memory_state], dim=1)
        else:
            inputs_embeds = torch.cat([inputs_embeds, memory_state], dim=1)
        
        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], dtype=inputs_embeds.dtype)
            if kwargs.get('prev_attn_mask') is not None:
                prev_seg_attn_mask = self.pad_prev_seg_attn_mask(kwargs['prev_attn_mask'], dtype=inputs_embeds.dtype)
                seg_kwargs['attention_mask'] = torch.cat([prev_seg_attn_mask, seg_kwargs['attention_mask']], dim=-1)
            if 'prev_attn_mask' in seg_kwargs:
                seg_kwargs.pop('prev_attn_mask')
        seg_kwargs['output_hidden_states'] = True

        if self.wrap_pos:
            num_pos_embs = self.model.transformer.wpe.weight.shape[0]
            ordinary_pos = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
            write_pos = torch.arange(num_pos_embs - self.num_mem_tokens, num_pos_embs, dtype=torch.long, device=input_ids.device)
            seg_kwargs['position_ids'] = torch.cat([
                ordinary_pos, 
                write_pos
            ]).long().unsqueeze(0)
        return seg_kwargs

    

    def pad_attention_mask(self, attention_mask, dtype=float):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            shape = list(attention_mask.shape)
            if len(shape) == 4:

                shape[-1] += self.num_mem_tokens + self.use_sink
                shape[-2] += self.num_mem_tokens + self.use_sink
                mask = torch.ones(*shape, dtype=dtype).to(attention_mask.device)
                mask[..., int(self.use_sink):-self.num_mem_tokens, int(self.use_sink):-self.num_mem_tokens] = attention_mask
                if self.use_sink:
                    mask[..., 0, 1:] = 0
                mask[..., :-self.num_mem_tokens, -self.num_mem_tokens:] = 0
                # mask = torch.tril(mask)
                if not os.environ.get("NOT_INVERT_ATTN_MASK"):
                    mask = invert_attn_mask(mask, dtype)
            else: 
                shape[-1] += self.num_mem_tokens + self.use_sink
                mask = torch.ones(*shape, dtype=dtype).to(attention_mask.device)
                mask[..., int(self.use_sink):-self.num_mem_tokens] = attention_mask
            return mask.to(dtype)

    def pad_prev_seg_attn_mask(self, prev_seg_attn_mask, dtype=float):
        if self.num_mem_tokens in {0, None}:
            return prev_seg_attn_mask
        else:
            shape = list(prev_seg_attn_mask.shape)
            if len(shape) == 4:
                shape[-2] += self.num_mem_tokens + self.use_sink
                mask = torch.ones(*shape, dtype=dtype).to(prev_seg_attn_mask.device)
                mask[..., int(self.use_sink):-self.num_mem_tokens, :] = prev_seg_attn_mask
                if self.use_sink:
                    mask[..., 0, :] = 0
                if not os.environ.get("NOT_INVERT_ATTN_MASK"):
                    mask = invert_attn_mask(mask, dtype)
            else: 
                mask = prev_seg_attn_mask
            return mask.to(dtype)
    
    def process_output(self, model_outputs, labels, labels_mask, **kwargs):
        

        if (self.num_mem_tokens not in {0, None}) and not self.RWKV_ARMT:
            out = CausalLMOutputWithCrossAttentions()
            out['logits'] = model_outputs.logits[:, int(self.use_sink):-self.num_mem_tokens]
            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, int(self.use_sink):-self.num_mem_tokens] for lh in model_outputs.hidden_states]
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
    
    def update_past_key_values_sw(self, past_key_values, window_size):
        past_key_values = past_key_values.to_legacy_cache()
        past_key_values = [
            [
                k_or_v[..., -(window_size+self.use_sink):, :]
                for k_or_v in seg_kv
            ]
            for seg_kv in past_key_values
        ]
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        return past_key_values
    
    def greedy_generate_sw(self, input_ids, attention_mask, prev_attn_mask, **generate_kwargs):
        self.generate_mode(True)
        window_size = generate_kwargs['window_size']
        max_new_tokens = generate_kwargs['max_new_tokens']
        past_key_values = self.update_past_key_values_sw(generate_kwargs['past_key_values'], window_size)
        eos_token_id = generate_kwargs['eos_token_id']
        prev_attn_mask_2d = prev_attn_mask.clone()
        attention_mask_2d = attention_mask.clone()
        
        attention_mask = attn_mask_to_4d(attention_mask, upper=False, query_len=attention_mask.size(-1))
        prev_attn_mask = attn_mask_to_4d(prev_attn_mask, upper=True, query_len=attention_mask.size(-1))
        seg_kwargs = self.process_input(input_ids=input_ids, attention_mask=attention_mask, prev_attn_mask=prev_attn_mask, past_key_values=past_key_values)
        seg_kwargs['inputs_embeds'] = seg_kwargs['inputs_embeds'][..., :-self.num_mem_tokens, :]
        seg_kwargs['attention_mask'] = seg_kwargs['attention_mask'][..., :-self.num_mem_tokens, :-self.num_mem_tokens]
        outputs = self.model(**seg_kwargs, use_cache=True)
        
        next_token_logits = outputs.logits[:, -1, :]

        past_key_values = outputs.past_key_values
        past_key_values = self.update_past_key_values_sw(past_key_values, window_size)

        generated_ids = None
        sw_attention_mask = torch.cat([prev_attn_mask_2d, torch.ones(attention_mask_2d.size(0), 1).to(prev_attn_mask_2d.device), attention_mask_2d], dim=-1)

        for i in range(max_new_tokens):
            # print(next_token_logits[..., :5])
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            if generated_ids is not None:
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            else:
                generated_ids = next_token_id
            next_input = next_token_id
            
            sw_attention_mask = torch.cat([sw_attention_mask, torch.ones_like(next_token_id).to(sw_attention_mask.device)], dim=-1)[..., -window_size-1-self.use_sink:]
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_input,
                    attention_mask=sw_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=torch.full((1,), window_size + i + input_ids.size(-1) + self.use_sink).to(input_ids.device)
                )
                past_key_values = self.update_past_key_values_sw(outputs.past_key_values, window_size)
                next_token_logits = outputs.logits[:, -1, :]
                
                if (next_token_id[:, 0] == eos_token_id).all():
                    break
        self.generate_mode(False)
        return generated_ids
            


class AssociativeRecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)
    
    

    def process_segment(self, segment_kwargs, next_seg_len=None):
        sliding_window = self.rmt_config['sliding_window'] if 'sliding_window' in self.rmt_config else False
        attend_to_previous_input = self.rmt_config['attend_to_previous_input'] if 'attend_to_previous_input' in self.rmt_config else False
        attn_mask = segment_kwargs['attention_mask']
        seg_len = segment_kwargs['input_ids'].size(-1)

        segment_kwargs['use_cache'] = sliding_window
        if segment_kwargs.get('past_key_values') is None:
            segment_kwargs['past_key_values'] = None
        if segment_kwargs.get('prev_attn_mask') is None:
            segment_kwargs['prev_attn_mask'] = None
        segment_kwargs['zero_mem'] = False
        if sliding_window or attend_to_previous_input:
            segment_kwargs['attention_mask'] = attn_mask_to_4d(attn_mask, upper=False, query_len=seg_len)
        
        
        num_mem_tokens = self.memory_cell.num_mem_tokens
        cell_out = self.memory_cell(**segment_kwargs)
        state = cell_out.get('state')
        if (sliding_window or attend_to_previous_input) and next_seg_len is not None:
            prev_attn_mask = attn_mask_to_4d(attn_mask, upper=True, query_len=next_seg_len)
        else: 
            prev_attn_mask = None
        if sliding_window:
            past_key_values = [
                [
                    k_or_v[..., -(num_mem_tokens+seg_len):k_or_v.size(-2)-num_mem_tokens, :].detach() 
                    for k_or_v in seg_kv
                ]
                for seg_kv in cell_out['past_key_values']
            ]
            if not isinstance(cell_out['past_key_values'], tuple) and not isinstance(cell_out['past_key_values'], list):
                past_key_values = cell_out['past_key_values'].from_legacy_cache(past_key_values)
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        else: 
            past_key_values = None
        next_segment_kwargs = dict()
        next_segment_kwargs['use_cache'] = sliding_window
        next_segment_kwargs['past_key_values'] = past_key_values
        next_segment_kwargs['prev_attn_mask'] = prev_attn_mask
        next_segment_kwargs['zero_mem'] = False
        if state is not None:
            next_segment_kwargs['state'] = state
        return cell_out, next_segment_kwargs
    
    def forward(self, 
                input_ids, 
                labels=None, 
                labels_mask=None, 
                inputs_embeds=None, 
                attention_mask=None, 
                output_attentions=None, 
                output_hidden_states=None,
                input_segmented=False,
                output_only_last_segment=False,
                ):
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
        self.memory_cell.zero_mem()
        next_seg_kwargs = dict()
        for seg_num, segment in enumerate(segmented):
            if seg_num != len(segmented) - 1:
                next_seg_len = segmented[seg_num + 1]['input_ids'].size(-1)
            else:
                next_seg_len = None
            cell_out, next_seg_kwargs = self.process_segment(dict(**segment, **next_seg_kwargs), next_seg_len=next_seg_len)
            if (not output_only_last_segment) or (seg_num == len(segmented) - 1):
                cell_outputs.append(cell_out)

        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states)
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
        if labels is not None:
            labels = labels[:, -full_logits.size(1):]
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                labels_mask = labels_mask[:, -full_logits.size(1):]
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]
                
            out['loss'] = loss_fct(flat_logits, flat_labels)
        else:
            out['loss'] = 0 
        if ('HF_Trainer' not in os.environ) or not os.environ['HF_Trainer']:
            out['ce_loss'] = out['loss']
        
        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states
        if ('HF_Trainer' not in os.environ) or not os.environ['HF_Trainer']:
            for seg_num, o in enumerate(cell_outputs):
                for key, value in o.items():
                    if any([sk in key for sk in segment_keys]):
                        out[f'{key}_{seg_num}'] = value

        remainders = []
        n_updates = []
        act_on = self.rmt_config['act_on'] if 'act_on' in self.rmt_config else False
        if act_on:

            for layer in self.memory_cell.layers:
                remainders.append(layer.remainders / layer.segments_passed)
                n_updates.append(layer.n_updates / layer.segments_passed)
            remainders = torch.mean(torch.stack(remainders, dim=0))
            n_updates = torch.mean(torch.stack(n_updates, dim=0))
            out['n_updates'] = n_updates.detach().cpu()
            out['remainders'] = remainders.detach().cpu()
            time_penalty = self.rmt_config['time_penalty']
            out['loss'] = out['loss'] + time_penalty * remainders
        
        return out 
        
    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return True
        
        memory_state = memory_state.detach()
        return False
    
    def generate(self, input_ids, attention_mask, **generate_kwargs):
        self.memory_cell.zero_mem()
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)
        next_seg_kwargs = dict()
        for seg_num, segment in enumerate(segmented[:-1]):
            next_seg_len = segmented[seg_num + 1]['input_ids'].size(-1)
            _, next_seg_kwargs = self.process_segment(dict(**segment, **next_seg_kwargs), next_seg_len=next_seg_len)
        
        final_segment = segmented[-1]
        assert next_seg_kwargs.get('past_key_values') is None or isinstance(next_seg_kwargs.get('past_key_values'), Cache), "Sliding Window generation is not implemented for legacy cache"
        if next_seg_kwargs.get('past_key_values') is not None:
            prev_attn_mask = segmented[-2]['attention_mask']
            legacy_cache = next_seg_kwargs['past_key_values'].to_legacy_cache()
            seg_len = segmented[-2]['input_ids'].size(-1)
            cache = DynamicCache().from_legacy_cache(legacy_cache)
            generate_kwargs['past_key_values'] = cache
            generate_kwargs['window_size'] = seg_len
            final_segment['prev_attn_mask'] = prev_attn_mask
            out = self.memory_cell.greedy_generate_sw(**final_segment, **generate_kwargs)
            return out
        else:
            out = self.memory_cell.generate(**final_segment, **generate_kwargs)
            return out