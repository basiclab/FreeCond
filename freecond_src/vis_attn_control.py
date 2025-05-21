import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union, Tuple, List, Callable, Dict

from torchvision.utils import save_image
from einops import rearrange, repeat
from collections import defaultdict

# Align to crossattention forward
class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, parent_module,input_tokens, q, k, v, mask, is_cross, place_in_unet, num_heads,scale, **kwargs):

        out = self.forward(parent_module,input_tokens, q, k, v, mask, is_cross, place_in_unet, num_heads, scale, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out
    # Aling to _attention in cross attention
    def forward(self, parent_module,input_tokens, q, k, v, mask, is_cross, place_in_unet, num_heads, scale, **kwargs):
        #print(f"Layer={self.cur_att_layer}, is_cross={is_cross}, place_in_unet={place_in_unet}, num_heads={num_heads}, qshape={q.shape},vshape={v.shape}")
        sim = torch.einsum('b i d, b j d -> b i j', q, k)*scale
        soft_sim = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', soft_sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        #
        #if is_cross:
        #    return torch.zeros_like(out)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class VISAttentionControl(AttentionBase):
    def __init__(self, start_step=0, end_step=50, start_layer=0, end_layer=16, layer_idx=None, step_idx=None, total_steps=50,
                  latent_mask=None, vis_cross=False, vis_self=False, vis_cross_token=False, downsample=64):
        super().__init__()
        self.layer_idx = list(range(start_layer, end_layer))
        self.step_idx = list(range(start_step, end_step))
        self.vis_self=vis_self
        self.vis_cross=vis_cross
        self.vis_cross_token=vis_cross_token
        print("Activating VIS Attention Control...")
        print("# Visualize cross attention (1, 3, 5, 7... layers) =", vis_cross)
        print("# Visualize self attention (0, 2, 4, 6 ... layers)=", vis_self)
        self.cross_atn_dict={"u":defaultdict(dict),
                             "c":defaultdict(dict)}
        self.cross_atn_dict_unsoft={"u":defaultdict(dict),
                             "c":defaultdict(dict)}
        self.self_atn_dict={"u":defaultdict(dict),
                             "c":defaultdict(dict)}
        self.cross_token_q_dict={"u":defaultdict(dict),
                        "c":defaultdict(dict)}
        self.cross_token_k_dict={"u":defaultdict(dict),
                             "c":defaultdict(dict)}
        self.cross_token_v_dict={"u":defaultdict(dict),
                        "c":defaultdict(dict)}
    # Perforem lps attention control (contextual token reduction)
    def vis_self_attn(self, parent_module,input_tokens, q, k, v, latent_mask, is_cross, place_in_unet, num_heads, scale, indicator="u", **kwargs):
        
        #print(f"Layer={self.cur_att_layer}, is_cross={is_cross}, place_in_unet={place_in_unet}, num_heads={num_heads}, qshape={q.shape},vshape={v.shape}")
        sim = torch.einsum('b i d, b j d -> b i j', q, k)*scale
        soft_sim = sim.softmax(dim=-1)
        self.self_atn_dict[indicator][self.cur_step][self.cur_att_layer]=soft_sim.cpu()
        out = torch.einsum('b i j, b j d -> b i d', soft_sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        #
        #if is_cross:
        #    return torch.zeros_like(out)
        return out
    def vis_cross_attn(self, parent_module,input_tokens, q, k, v, latent_mask, is_cross, place_in_unet, num_heads, scale, indicator="u", **kwargs):
        
        #print(f"Layer={self.cur_att_layer}, is_cross={is_cross}, place_in_unet={place_in_unet}, num_heads={num_heads}, qshape={q.shape},vshape={v.shape}")
        sim = torch.einsum('b i d, b j d -> b i j', q, k)*scale
        soft_sim = sim.softmax(dim=-1)
        # print(f"saving to [{indicator}][{self.cur_step}][{self.cur_att_layer}]")
        self.cross_atn_dict_unsoft[indicator][self.cur_step][self.cur_att_layer]=sim.cpu()
        self.cross_atn_dict[indicator][self.cur_step][self.cur_att_layer]=soft_sim.cpu()
        out = torch.einsum('b i j, b j d -> b i d', soft_sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        if self.vis_cross_token:
            self.cross_token_q_dict[indicator][self.cur_step][self.cur_att_layer] = q.cpu()
            self.cross_token_k_dict[indicator][self.cur_step][self.cur_att_layer] = k.cpu()
            self.cross_token_v_dict[indicator][self.cur_step][self.cur_att_layer] = v.cpu()
        #
        #if is_cross:
        #    return torch.zeros_like(out)
        return out
    def forward(self, parent_module,input_tokens, q, k, v, latent_mask, is_cross, place_in_unet, num_heads, scale, **kwargs):
        """
        Attention forward function
        """
        
        # !!! is_cross: using original attention forward
        # self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx :
        if self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            # Performing CA or normal SA
            return super().forward(parent_module,input_tokens, q, k, v, latent_mask, is_cross, place_in_unet, num_heads,scale, **kwargs)
        if is_cross and self.vis_cross:
            # cross attention on uncondition branch and conditional branch (prompt)
            inu, inc=input_tokens.chunk(2)
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            
            out_u = self.vis_cross_attn(parent_module,inu, qu, ku, vu, latent_mask, is_cross, place_in_unet, num_heads, scale, indicator="u", **kwargs)
            out_c = self.vis_cross_attn(parent_module,inc, qc, kc, vc, latent_mask, is_cross, place_in_unet, num_heads, scale, indicator="c", **kwargs)

            out = torch.cat([out_u, out_c], dim=0)
        elif not is_cross and self.vis_self:
            # self attention on uncondition branch and conditional branch (prompt)
            inu, inc=input_tokens.chunk(2)
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            
            out_u = self.vis_self_attn(parent_module,inu, qu, ku, vu, latent_mask, is_cross, place_in_unet, num_heads, scale, indicator="u", **kwargs)
            out_c = self.vis_self_attn(parent_module,inc, qc, kc, vc, latent_mask, is_cross, place_in_unet, num_heads, scale, indicator="c", **kwargs)

            out = torch.cat([out_u, out_c], dim=0)
        else:
            # Performing CA or normal SA
            return super().forward(parent_module,input_tokens, q, k, v, latent_mask, is_cross, place_in_unet, num_heads,scale, **kwargs)
        return out



def regiter_attention_editor_diffusers(model, editor: AttentionBase, merge=False):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    For facilating QKV modification we rewrite the
    """
    # Align to crossattention forward, self is the cross_attention module
    def ca_forward(self, place_in_unet):
        # Align to crossattention forward
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)

            # the only difference
            # Aling to _attention in cross attention

            out = editor(
                self, x, q, k, v, mask, is_cross, place_in_unet,
                self.heads, scale=self.scale)


            return to_out(out)

        return forward
    # !!!
    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count
