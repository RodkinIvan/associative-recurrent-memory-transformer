import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.cache_utils import Cache, DynamicCache
import torch.nn.functional as F
import os
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from modeling_amt.utils import DPFP

from modeling_amt.utils import attn_mask_to_4d, invert_attn_mask

@dataclass
class ARMTOutput(ModelOutput):
    """
    Custom output format for ARMT with all necessary fields.
    This replaces Munch in the original implementation.
    """
    logits: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: tuple = None
    past_key_values: tuple = None
    remainders: torch.FloatTensor = None
    n_updates: torch.FloatTensor = None
    ce_loss: torch.FloatTensor = None

# Import optimized cross-entropy loss
try:
    from cut_cross_entropy import linear_cross_entropy
    CUT_CROSS_ENTROPY_AVAILABLE = True
except ImportError:
    CUT_CROSS_ENTROPY_AVAILABLE = False
    print("Warning: cut_cross_entropy not available, falling back to standard CrossEntropyLoss")

from modeling_amt.act_utils import ACT_basic, gen_timing_signal, ACTForWholeARMT, ACT_transformer, ACT_constant_depth, ACTForWholeARMT_constant_depth
try:
    from baselines.rwkv.language_modeling import RWKVModel
    RWKV_imported = True
except ImportError:
    print("*** Can't import RWKV model ***")
    RWKV_imported = False




class AssociativeLayerWrapper(torch.nn.Module):

    def __init__(self, layer, d_model,  num_mem_tokens, d_mem, n_heads=1, correction=True, info=None, use_denom=True, gating=False) -> None:
        super().__init__()
        self.info = info
        self.seg_num = 0
        self.d_model = d_model
        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.gating = gating
        nu = 3
        self.d_key = 2 * nu * d_mem

        assert self.d_mem % n_heads == 0 and self.d_model % n_heads == 0

        self.phi = DPFP(nu)
        # self.d_key = d_mem
        # self.phi = torch.nn.Identity()

        self.use_denom = use_denom

        # Get the proper dtype from the layer
        layer_dtype = next(layer.parameters()).dtype
        
        self.W_mq = torch.nn.Linear(d_model, d_mem, bias=False, dtype=layer_dtype)
        # torch.nn.init.zeros_(self.W_mq.weight)
        self.W_mk = torch.nn.Linear(d_model, d_mem, bias=False, dtype=layer_dtype)
        self.W_mv = torch.nn.Linear(d_model, d_model, bias=False, dtype=layer_dtype)
        if gating:
            self.W_mb = torch.nn.Linear(d_model, d_model, dtype=layer_dtype)
        else:
            self.W_mb = torch.nn.Linear(d_model, n_heads, dtype=layer_dtype)
        torch.nn.init.zeros_(self.W_mv.weight)
        s = 1/math.sqrt(d_model)
        # torch.nn.init.uniform_(self.W_mq.weight, -s, s)
        # torch.nn.init.uniform_(self.W_mk.weight, -s, s)
        # torch.nn.init.uniform_(self.W_mb.weight, -s, s)


        # self.ln = torch.nn.LayerNorm(d_model)

        self.layer = layer
        
        self.generate_mode = False
        self.first_seg = True
        self.correction = correction
        
        self.zero_mem()

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped layer if not found in wrapper."""
        # First let nn.Module try to find the attribute (in _modules, _parameters, _buffers)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Then delegate to the wrapped layer
        try:
            layer = super().__getattr__('layer')
            return getattr(layer, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _to_heads(self, x):
        bsz, seq_len, d_model = x.shape
        x = x.reshape(bsz, seq_len, self.n_heads, d_model // self.n_heads)
        x = x.permute(0, 2, 1, 3)
        return x
    
    def _from_heads(self, x):
        bsz, n_heads, seq_len, d_head = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * d_head)
        return x
    def associate(self, hidden_states):
        bsz, seq_len, d_model = hidden_states.shape

        self.W_mem = self.W_mem.to(hidden_states.device)
        if self.use_denom:
            self.z = self.z.to(hidden_states.device)

        q = self._to_heads(self.W_mq(hidden_states))
        mq = self.phi(q) # (bsz, n_heads, seq_len, 2 * d_head * nu)
        mq = F.normalize(mq, dim=-1, p=2.0)
        # crutch for dataparallel
        # mq += 0 * self.W_mb(hidden_states).sum() * self.W_mk(hidden_states).sum() * self.W_mv(hidden_states).sum()
        num = torch.einsum('ihjk,ihkt->ihjt', mq, self.W_mem)
        if self.use_denom:
            denom = torch.einsum("ihk,ihjk->ihj", self.z, mq)[..., None] + 1e-5
            hidden_states = num / denom # (bsz, n_heads, seq_len, d_model // n_heads)
        else:
            hidden_states = num
        hidden_states = self._from_heads(hidden_states)
        return hidden_states
    
    def forward(self, hidden_states, *args, **kwargs):
        if not self.first_seg:
            hidden_states = self.associate(
                # self.ln(
                    hidden_states
                # )
            ) + hidden_states
        out = self.layer(hidden_states, *args, **kwargs)
        if not self.generate_mode:
            # The layer output contains hidden states, not logits
            # For transformer layers, the output is typically the hidden states
            if isinstance(out, tuple):
                mem_tokens = out[0][:, -self.num_mem_tokens:]
            else:
                mem_tokens = out[:, -self.num_mem_tokens:]

            self.update_mem(mem_tokens)
        return out
    
    def forward_no_update(self, hidden_states, *args, **kwargs):
        if not self.first_seg:
            hidden_states = self.associate(
                # self.ln(
                    hidden_states
                # )
            )+ hidden_states
        out = self.layer(hidden_states, *args, **kwargs)
        return out
    
    def forward_no_update(self, hidden_states, *args, **kwargs):
        if not self.first_seg:
            hidden_states = self.associate(
                # self.ln(
                    hidden_states
                # )
            ) + hidden_states
        out = self.layer(hidden_states, *args, **kwargs)
        return out

    def update_mem(self, mem_tokens):

        self.W_mem = self.W_mem.to(mem_tokens.device)
        if self.use_denom:
            self.z = self.z.to(mem_tokens.device)
        k = self._to_heads(self.W_mk(mem_tokens))
        mk = self.phi(k)
        mk = F.normalize(mk, dim=-1, p=2.0)

        new_mv = self._to_heads(self.W_mv(mem_tokens)) # (bsz, n_heads, num_mem_tokens, d_model)
        if not self.first_seg:
            num = torch.einsum('ihjk,ihkt->ihjt', mk, self.W_mem)
            if self.use_denom:
                denom = torch.einsum("ihj,ihkj->ihk", self.z, mk)[..., None] + 1e-5
                prev_mv = num / denom
                if self.correction:
                    new_info_coef = (1 - denom / (torch.linalg.norm(mk, dim=-1) ** 2)[..., None])
                    new_info_coef = torch.clip(new_info_coef, 0, 1).detach()
                else:
                    new_info_coef = 1
            else:
                prev_mv = num
        else: 
            prev_mv = torch.zeros_like(new_mv, device=new_mv.device)
            new_info_coef = 1
        
        mv = new_mv - prev_mv

        # new_norm = torch.linalg.norm(new_mv, dim=-1)
        # old_norm = torch.linalg.norm(prev_mv, dim=-1)
        # new_info_coef = torch.clip(1 - old_norm / (new_norm + 1e-5), -10, 10)[..., None].detach()
        # new_info_coef = 1 - denom

        mb = self._to_heads(torch.sigmoid(self.W_mb(mem_tokens)))

        einop = f"ihjk,ihjt,ihj{'t' if self.gating else 'x'}->ihkt"
        associations =  torch.einsum(einop, mk, mv, mb) # (bsz, n_heads, d_mem, d_model)

        self.W_mem = self.W_mem + associations

        if self.use_denom:
            self.z = self.z + (new_info_coef*mk).sum(dim=-2)
        # self.z = self.z + (new_info_coef*mb[..., None]*mk).sum(dim=1)
        self.seg_num += 1
        self.first_seg = False

    def freeze_mem(self):
        self.W_mb.weight.requires_grad = False
        self.W_mb.bias.requires_grad = False
        self.W_mq.weight.requires_grad = False
        self.W_mk.weight.requires_grad = False
        self.W_mv.weight.requires_grad = False

    def zero_mem(self):
        self.first_seg = True
        # Get the proper dtype from the layer parameters
        layer_dtype = next(self.layer.parameters()).dtype
        self.W_mem = torch.zeros(1, self.n_heads, self.d_key // self.n_heads, self.d_model // self.n_heads, dtype=layer_dtype)
        self.W_mem.requires_grad_(False)
        if self.use_denom:
            self.z = torch.zeros(1, self.n_heads, self.d_key // self.n_heads, dtype=layer_dtype)
            self.z.requires_grad_(False)
        self.seg_num = 0

    def detach_mem(self):
        self.W_mem = self.W_mem.detach()
        if self.use_denom:
            self.z = self.z.detach()




class AdaptiveAssociativeLayerWrapper(AssociativeLayerWrapper):
    def __init__(self, 
                 layer, 
                 d_model, 
                 num_mem_tokens, 
                 d_mem, 
                 max_hop,
                 n_heads=1, 
                 correction=True, 
                 info=None, 
                 use_denom=True, 
                 gating=False,
                 constant_depth=False,
                 
                ) -> None:
        super().__init__(layer, d_model, num_mem_tokens, d_mem, n_heads, correction, info, use_denom, gating)
        self.act = ACT_basic(d_model) if not constant_depth else ACT_constant_depth()
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
        
        self.remainders = self.remainders + remainders.mean() # 1 - \sum(h_i); L' = L + tau * mean(remainders)
        self.n_updates = self.n_updates + n_updates.mean()
        self.segments_passed = self.segments_passed + 1
        return out
    
    def zero_mem(self):
        self.remainders = torch.zeros(1,)
        self.n_updates = torch.zeros(1,)
        self.segments_passed = torch.zeros(1,)
        return super().zero_mem()
    
    def detach_mem(self):
        self.remainders = torch.zeros(1,)
        self.n_updates = torch.zeros(1,)
        self.segments_passed = torch.zeros(1,)
        return super().detach_mem()



class AdaptiveAssociativeLayerWrapper2(AssociativeLayerWrapper):
    def __init__(self, 
                 layer, 
                 d_model, 
                 num_mem_tokens, 
                 d_mem, 
                 max_hop,
                 n_heads=1, 
                 correction=True, 
                 info=None, 
                 use_denom=True, 
                 gating=False,
                 act_format='linear',
                 noisy_halting=False,
                 constant_depth=False,
                ) -> None:
        super().__init__(layer, d_model, num_mem_tokens, d_mem, n_heads, correction, info, use_denom, gating)

        if act_format=='transformer':
            self.act = ACT_transformer(d_model)
        elif constant_depth:
            self.act = ACT_constant_depth()
        elif act_format == 'linear':
            self.act =  ACT_basic(d_model)
        else:
            raise NotImplemetedError

        self.depth = max_hop
        self.max_length = 1024

        self.noisy_halting = noisy_halting

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

        if self.noisy_halting:
            kwargs['noisy_halting'] = self.noisy_halting
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
            self.first_seg = False
        self.remainders = self.remainders + remainders.mean() # 1 - \sum(h_i); L' = L + tau * mean(remainders)
        self.n_updates = self.n_updates + n_updates.mean()
        self.segments_passed = self.segments_passed + 1
        return out

    
    def zero_mem(self):
        self.remainders = torch.zeros(1,)
        self.n_updates = torch.zeros(1,)
        self.segments_passed = torch.zeros(1,)
        return super().zero_mem()
    
    def detach_mem(self):
        self.remainders = torch.zeros(1,)
        self.n_updates = torch.zeros(1,)
        self.segments_passed = torch.zeros(1,)
        return super().detach_mem()


class AdaptiveAssociativeLayerWrapper(AssociativeLayerWrapper):
    def __init__(self, 
                 layer, 
                 d_model, 
                 num_mem_tokens, 
                 d_mem, 
                 max_hop,
                 n_heads=1, 
                 correction=True, 
                 info=None, 
                 use_denom=True, 
                 gating=False,
                 
                ) -> None:
        super().__init__(layer, d_model, num_mem_tokens, d_mem, n_heads, correction, info, use_denom, gating)
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
    


class AssociativeMemoryCell(torch.nn.Module):
    def __init__(self, 
                 base_model, 
                 num_mem_tokens, 
                 d_mem,
                 layers_attr: str = 'model.layers', 
                 wrap_pos=False, 
                 correction=True, 
                 n_heads=1, 
                 use_denom=True, 
                 gating=False, 
                 freeze_mem=False,
                 act_on=False,
                 max_hop=4,
                 act_type='layer',
                 act_format='linear',
                 noisy_halting=False,
                 constant_depth=False,
                 attend_to_previous_input=False,
                 use_sink=False,
                 **rmt_config
        ):
        super().__init__()
        self.model = base_model
        
        self.attend_to_previous_input = attend_to_previous_input
        self.previous_input = None
        self.use_sink = use_sink
        
        self.RWKV_ARMT = isinstance(self.model, RWKVModel) if RWKV_imported else False

        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem
        self.d_model = base_model.get_input_embeddings().embedding_dim
        self.W_mem = []

        self.constant_depth = constant_depth

        self.layers_attrs = layers_attr.split('.')

        def _get_layers_from_model(model_root):
            layers_obj = model_root
            for attr in self.layers_attrs:
                layers_obj = getattr(layers_obj, attr)
            return layers_obj

        layers = _get_layers_from_model(self.model)
        
        for i in range(len(layers)):
            kw = dict(
                layer=layers[i], 
                d_model=self.d_model, 
                num_mem_tokens=self.num_mem_tokens, 
                d_mem=self.d_mem,
                correction=correction,
                info={'layer': i},
                n_heads=n_heads,
                use_denom=use_denom,
                gating=gating,
            )
            if act_on and act_type != 'model':
                kw['act_format'] = act_format
            if act_on and act_type == 'model' and act_format != 'linear':
                raise NotImplementedError
            if act_on and (act_type != 'model'):
                kw['max_hop'] = max_hop
                kw['constant_depth'] = self.constant_depth
                kw['act_format'] = act_format
            if act_on and noisy_halting:
                kw['noisy_halting'] = noisy_halting
            if not act_on:
                layers[i] = AssociativeLayerWrapper(**kw)
            elif act_type == 'associative':
                layers[i] = AdaptiveAssociativeLayerWrapper(**kw)
            elif act_type == 'layer':
                layers[i] = AdaptiveAssociativeLayerWrapper2(**kw)
            elif act_type == 'model':
                layers[i] = AssociativeLayerWrapper(**kw)
            else:
                raise f'Unknown ACT type: {act_type}'

        if act_type == 'model':
            self.act = ACTForWholeARMT(self.d_model) if not self.constant_depth else ACTForWholeARMT_constant_depth()
            self.depth = max_hop
            self.max_length = 1024
            self.timing_signal = gen_timing_signal(self.max_length, self.d_model)
            self.position_signal = gen_timing_signal(self.depth, self.d_model)
        self.act_type = act_type

        self.create_memory(num_mem_tokens)
        self.wrap_pos = wrap_pos
        self.act_on = act_on
        if wrap_pos:
            self.wrap_positional_embeddings(num_mem_tokens)
        
        if freeze_mem:
            for layer in _get_layers_from_model(self.model):
                layer.freeze_mem()

        # Expose a resolver without registering layers as a submodule to avoid shared tensor aliases
        self.get_layers = lambda: _get_layers_from_model(self.model)
    
    def generate_mode(self, is_on):
        for layer in self.get_layers():
            layer.generate_mode = is_on
    
    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim =  getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim), device=embeddings.weight.data.device, dtype=embeddings.weight.data.dtype) * embeddings.weight.data.std()

        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))
        if self.use_sink:
            self.sink = torch.nn.Parameter(torch.randn((1, memory_dim), device=embeddings.weight.data.device, dtype=embeddings.weight.data.dtype), requires_grad=True)


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
        for layer in self.get_layers():
            layer.zero_mem()
        self.previous_input = None
    
    def detach_mem(self):
        for layer in self.get_layers():
            layer.detach_mem()
            pass

    def forward(self, input_ids, labels=None, labels_mask=None, zero_mem=False, attention_mask=None, **kwargs):
        if self.act_type != 'model':
            out = self.forward_with_update(input_ids, labels, labels_mask, zero_mem, attention_mask=attention_mask, **kwargs)
        else:
            seg_kwargs = self.process_input(input_ids=input_ids, 
                                            labels=labels, 
                                            labels_mask=labels_mask, 
                                            zero_mem=zero_mem, 
                                            attention_mask=attention_mask, 
                                            **kwargs
                                        )
            out = self.gptneox_forward_act(**seg_kwargs)
            out = self.process_output(out, labels=labels, labels_mask=labels_mask)
        return out

    def forward_with_update(self, input_ids, labels=None, labels_mask=None, zero_mem=False, **kwargs):
        current_input_ids = input_ids.clone()
        if self.attend_to_previous_input and self.previous_input is not None:
            input_ids = torch.cat([self.previous_input, input_ids], dim=1)
        
        if zero_mem:
            self.zero_mem()

        seg_kwargs = self.process_input(input_ids, **kwargs)
        
        layers = self.get_layers()
        if self.RWKV_ARMT and not layers[0].generate_mode:
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
            out = ARMTOutput(**{k: torch.clone(t) if isinstance(t, torch.Tensor) else t for k, t in out.items()})
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
            labels = labels[..., 1:].contiguous()
            flat_labels = labels.view(-1)
            
            if labels_mask is not None:
                flat_mask = labels_mask[..., :-1].contiguous().view(-1)
                flat_labels = flat_labels[flat_mask]
            
            # Use optimized linear cross-entropy if available
            if CUT_CROSS_ENTROPY_AVAILABLE and hasattr(self.model, 'embed_out'):
                # Get hidden states from the last layer (before LM head)
                if 'hidden_states' in model_outputs and model_outputs.hidden_states is not None:
                    # Use the last hidden state
                    hidden_states = model_outputs.hidden_states[-1]
                    # Remove memory tokens from hidden states
                    if self.num_mem_tokens not in {0, None}:
                        hidden_states = hidden_states[:, int(self.use_sink):-self.num_mem_tokens]
                    # Shift for next token prediction
                    hidden_states = hidden_states[..., :-1, :].contiguous()
                    flat_hidden_states = hidden_states.view(-1, hidden_states.size(-1))
                    
                    if labels_mask is not None:
                        flat_hidden_states = flat_hidden_states[flat_mask]
                    
                    # Get LM head weights
                    lm_head_weights = self.model.embed_out.weight  # Shape: (vocab_size, hidden_size)
                    
                    # Use linear_cross_entropy with hidden states and LM head weights
                    ce_loss = linear_cross_entropy(
                        flat_hidden_states,  # embeddings
                        lm_head_weights,    # classifier weights
                        flat_labels,        # targets
                        reduction='sum'
                    )
                else:
                    # Fallback to standard approach if hidden states not available
                    logits = out['logits'][..., :-1, :].contiguous()
                    flat_logits = logits.view(-1, logits.size(-1))
                    if labels_mask is not None:
                        flat_logits = flat_logits[flat_mask]
                    ce_loss_fn = CrossEntropyLoss(reduction='sum')
                    ce_loss = ce_loss_fn(flat_logits, flat_labels)
            else:
                # Fallback to standard CrossEntropyLoss
                logits = out['logits'][..., :-1, :].contiguous()
                flat_logits = logits.view(-1, logits.size(-1))
                if labels_mask is not None:
                    flat_logits = flat_logits[flat_mask]
                ce_loss_fn = CrossEntropyLoss(reduction='sum')
                ce_loss = ce_loss_fn(flat_logits, flat_labels)
            
            if labels_mask is not None:
                denom = labels_mask[..., :-1].contiguous().view(-1).sum()
            else:
                denom = (flat_labels != -100).sum()
            denom = torch.clamp(denom, min=1)
            out['ce_loss'] = ce_loss / denom

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
            

    def apply_layers(self, hidden_states, causal_mask, position_ids, cache_position, position_embeddings, update_mem=True):
        if not update_mem:
            tmp = []
            for i in range(len(self.layers)):
                tmp.append(self.layers[i].forward)
                self.layers[i].forward = self.layers[i].forward_no_update

        for layer in self.get_layers():
            hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )[0]

        if not update_mem:
            for i, layer in enumerate(self.get_layers()):
                layer.forward = tmp[i]
        return hidden_states
        
    
    def gptneox_forward_act(self, inputs_embeds, labels=None, labels_mask=None, zero_mem=False, attention_mask=None, **kwargs):
            
            drop = self.model.gpt_neox.emb_dropout
            hidden_states = drop(inputs_embeds)
            seq_length = hidden_states.shape[1]
            cache_position = torch.arange(0, seq_length, device=hidden_states.device)
            position_ids = cache_position.unsqueeze(0)

            position_embeddings = self.model.gpt_neox.rotary_emb(hidden_states, position_ids)
            causal_mask = self.model.gpt_neox._update_causal_mask(
                attention_mask, hidden_states, cache_position, None, False
            )

            out, (remainders, n_updates) = self.act(
                state=hidden_states,
                inputs=hidden_states,
                fn_no_update=lambda *args, **kwargs: self.apply_layers(*args, **kwargs, update_mem=False),
                fn_update=self.apply_layers,
                time_enc=self.timing_signal,
                pos_enc=self.position_signal,
                max_hop=self.depth,
                causal_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )
            hidden_states = self.model.gpt_neox.final_layer_norm(out)

            lm_logits = self.model.embed_out(hidden_states)
            return ARMTOutput(logits=lm_logits, n_updates=n_updates, remainders=remainders)

class AssociativeRecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs
        self.last_state = None

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
        
        if 'state' in segment_kwargs and segment_kwargs['state'] is None:
            segment_kwargs.pop('state')
        
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
                use_previous_batch_state=torch.zeros(1),
                num_items_in_batch=None,  # Added to handle HF Trainer compatibility
                **kwargs  # Added to handle any other unexpected kwargs
                ):
        if input_segmented:
            # Support passing already-segmented inputs either as:
            # - lists/tuples of tensors: [(bsz, seg_len_i), ...]  [new behavior]
            def _is_seq(x):
                return isinstance(x, (list, tuple))

            # List/tuple of segments: segment_size is ignored.
            if input_ids is None and inputs_embeds is None:
                raise ValueError("When input_segmented=True, either input_ids or inputs_embeds must be provided.")
            if not (_is_seq(input_ids) or _is_seq(inputs_embeds)):
                raise TypeError(
                    "When input_segmented=True, input_ids/inputs_embeds must be provided as a list/tuple of segments "
                    "(each segment tensor shaped (bsz, seq_len) for input_ids or (bsz, seq_len, d) for inputs_embeds)."
                )
            if attention_mask is None:
                raise ValueError("When input_segmented=True, attention_mask must be a list/tuple with one tensor per segment.")

            n_segs = len(input_ids) if _is_seq(input_ids) else len(inputs_embeds)
            if _is_seq(input_ids) and len(input_ids) != n_segs:
                raise ValueError("input_ids must have one tensor per segment.")
            if _is_seq(inputs_embeds) and len(inputs_embeds) != n_segs:
                raise ValueError("inputs_embeds must have one tensor per segment.")
            if not _is_seq(attention_mask) or len(attention_mask) != n_segs:
                raise ValueError("attention_mask must be a list/tuple with the same number of segments as input_ids/inputs_embeds.")
            # labels / labels_mask can be provided either segmented (list/tuple) or concatenated (tensor).
            seg_lens = [am.size(-1) for am in attention_mask]
            if labels is not None and not (_is_seq(labels) or isinstance(labels, torch.Tensor)):
                raise TypeError("labels must be either a list/tuple of segments or a concatenated torch.Tensor when input_segmented=True.")
            if labels_mask is not None and not (_is_seq(labels_mask) or isinstance(labels_mask, torch.Tensor)):
                raise TypeError("labels_mask must be either a list/tuple of segments or a concatenated torch.Tensor when input_segmented=True.")

            labels_segs = None
            labels_mask_segs = None
            if isinstance(labels, torch.Tensor):
                if labels.size(1) != sum(seg_lens):
                    raise ValueError(
                        f"Concatenated labels must have seq_len={sum(seg_lens)} (sum of segment lengths), got {labels.size(1)}."
                    )
                labels_segs = list(torch.split(labels, seg_lens, dim=1))
            elif _is_seq(labels):
                if len(labels) != n_segs:
                    raise ValueError("labels must have the same number of segments as input_ids/inputs_embeds when input_segmented=True.")
                labels_segs = list(labels)

            if isinstance(labels_mask, torch.Tensor):
                if labels_mask.size(1) != sum(seg_lens):
                    raise ValueError(
                        f"Concatenated labels_mask must have seq_len={sum(seg_lens)} (sum of segment lengths), got {labels_mask.size(1)}."
                    )
                labels_mask_segs = list(torch.split(labels_mask, seg_lens, dim=1))
            elif _is_seq(labels_mask):
                if len(labels_mask) != n_segs:
                    raise ValueError("labels_mask must have the same number of segments as input_ids/inputs_embeds when input_segmented=True.")
                labels_mask_segs = list(labels_mask)

            segmented = []
            for i in range(n_segs):
                segmented.append(dict(
                    input_ids=input_ids[i] if _is_seq(input_ids) else None,
                    inputs_embeds=inputs_embeds[i] if _is_seq(inputs_embeds) else None,
                    attention_mask=attention_mask[i],
                    labels=labels_segs[i] if labels_segs is not None else None,
                    labels_mask=labels_mask_segs[i] if labels_mask_segs is not None else None,
                ))

            if labels_segs is not None and not isinstance(labels, torch.Tensor):
                labels = torch.cat(labels_segs, dim=1)
            if labels_mask_segs is not None and not isinstance(labels_mask, torch.Tensor):
                labels_mask = torch.cat(labels_mask_segs, dim=1)
        else:
            segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, labels_mask=labels_mask)
        
        cell_outputs = []
        if not use_previous_batch_state.all() or self.last_state is None:
            self.memory_cell.zero_mem()
            state = None
        else: 
            self.memory_cell.detach_mem()
            state = self.last_state
        next_seg_kwargs = dict(state=state)
        for seg_num, segment in enumerate(segmented):
            if seg_num != len(segmented) - 1:
                next_seg_ids = segmented[seg_num + 1].get('input_ids')
                next_seg_emb = segmented[seg_num + 1].get('inputs_embeds')
                next_seg_len = next_seg_ids.size(-1) if next_seg_ids is not None else next_seg_emb.size(-2)
            else:
                next_seg_len = None
            # Pass num_items_in_batch to segment processing
            segment_with_kwargs = dict(**segment, **next_seg_kwargs)
            if kwargs.get('num_items_in_batch') is not None:
                segment_with_kwargs['num_items_in_batch'] = kwargs['num_items_in_batch']
            cell_out, next_seg_kwargs = self.process_segment(segment_with_kwargs, next_seg_len=next_seg_len)
            if (not output_only_last_segment) or (seg_num == len(segmented) - 1):
                cell_outputs.append(cell_out)

        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states,
                                   num_items_in_batch=kwargs.get('num_items_in_batch'))
        
        if not self.training:
            self.memory_cell.zero_mem()
            self.last_state = None
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
        out = ARMTOutput()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        
        labels = kwargs.get('labels')
        if labels is not None:
            labels = labels[:, -full_logits.size(1):]
            shift_labels = labels[..., 1:].contiguous()
            flat_labels = shift_labels.view(-1)
            
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                labels_mask = labels_mask[:, -full_logits.size(1):]
                shift_mask = labels_mask[..., :-1].contiguous()
                flat_labels = flat_labels[shift_mask.view(-1)]
            
            # Use optimized linear cross-entropy if available
            if CUT_CROSS_ENTROPY_AVAILABLE and hasattr(self.memory_cell.model, 'embed_out'):
                # Get hidden states from the last segment
                if cell_outputs and 'hidden_states' in cell_outputs[-1] and cell_outputs[-1].hidden_states is not None:
                    # Concatenate hidden states from all segments
                    full_hidden_states = torch.cat([o.hidden_states[-1] for o in cell_outputs], dim=1)
                    # Shift for next token prediction
                    shift_hidden_states = full_hidden_states[..., :-1, :].contiguous()
                    flat_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))
                    
                    if labels_mask is not None:
                        flat_hidden_states = flat_hidden_states[shift_mask.view(-1)]
                    
                    # Get LM head weights
                    lm_head_weights = self.memory_cell.model.embed_out.weight  # Shape: (vocab_size, hidden_size)
                    
                    # Use linear_cross_entropy with hidden states and LM head weights
                    loss = linear_cross_entropy(
                        flat_hidden_states,  # embeddings
                        lm_head_weights,    # classifier weights
                        flat_labels,        # targets
                        reduction='sum'
                    )
                else:
                    # Fallback to standard approach if hidden states not available
                    shift_logits = full_logits[..., :-1, :].contiguous()
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                    if labels_mask is not None:
                        flat_logits = flat_logits[shift_mask.view(-1)]
                    loss_fct = CrossEntropyLoss(reduction='sum')
                    loss = loss_fct(flat_logits, flat_labels)
            else:
                # Fallback to standard CrossEntropyLoss
                shift_logits = full_logits[..., :-1, :].contiguous()
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                if labels_mask is not None:
                    flat_logits = flat_logits[shift_mask.view(-1)]
                loss_fct = CrossEntropyLoss(reduction='sum')
                loss = loss_fct(flat_logits, flat_labels)
            
            if labels_mask is not None:
                # Use the same mask used to filter flat logits/labels
                denom = labels_mask[..., :-1].contiguous().view(-1).sum()
            else:
                denom = (flat_labels != -100).sum()
            denom = torch.clamp(denom, min=1)
            out['loss'] = loss / denom
        else:
            out['loss'] = 0 
        if ('HF_Trainer' not in os.environ) or not os.environ['HF_Trainer']:
            out['ce_loss'] = out['loss']
        
        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            # Only process hidden_states if all cell outputs have them
            if all(hasattr(o, 'hidden_states') and o.hidden_states is not None for o in cell_outputs):
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
          if self.memory_cell.act_type != 'model':
            for layer in self.memory_cell.get_layers():
                remainders.append(layer.remainders / layer.segments_passed)
                n_updates.append(layer.n_updates / layer.segments_passed)
            remainders = torch.mean(torch.stack(remainders, dim=0))
            n_updates = torch.mean(torch.stack(n_updates, dim=0))
          else:
            remainders = torch.mean(torch.stack([o['remainders'] for o in cell_outputs], dim=0))
            n_updates = torch.mean(torch.stack([o['n_updates'] for o in cell_outputs], dim=0))
          out['n_updates'] = n_updates.detach().cpu()
          out['remainders'] = remainders.detach().cpu()
          time_penalty = self.rmt_config['time_penalty']
          out['loss'] = out['loss'] + time_penalty * remainders
        
        return out 
    
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
