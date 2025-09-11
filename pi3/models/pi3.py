import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope, AttentionRopeFP8
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg, dinov2_vits14_reg, dinov2_vitb14_reg
from huggingface_hub import PyTorchModelHubMixin

class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            global_merging: bool = False,
            merging: int = 0,
            merge_ratio: float = 0.9,
            use_fp8_attention: bool = False,
        ):
        super().__init__()

        # ----------------------
        #   Token Merging Config
        # ----------------------
        self.do_global_merging = global_merging
        self.merging = merging
        self.merge_ratio = merge_ratio

        # ----------------------
        #        Encoder
        # ----------------------
        if decoder_size == 'small':
            self.encoder = dinov2_vits14_reg(pretrained=True)
            self.patch_size = 14
        elif decoder_size in ['base']:
            self.encoder = dinov2_vitb14_reg(pretrained=False)
            self.patch_size = 14
        elif decoder_size in ['large']:
            self.encoder = dinov2_vitl14_reg(pretrained=False)
            self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        attn_impl = AttentionRopeFP8 if use_fp8_attention else FlashAttentionRope
        print(f"Using {attn_impl.__name__} for attention")
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=attn_impl,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        self.project_stud_to_teach = nn.Linear(2*384, 2*1024)

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*1024, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
            use_fp8_attention=use_fp8_attention,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*1024, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False,
            use_fp8_attention=use_fp8_attention,
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.is_distillation = False

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)


    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        # Pre-calculate merging functions if merging is enabled
        merging_functions = {}
        if self.do_global_merging and self.merging is not None:
            merging_functions = self._precalculate_merging_functions(hidden, N, H, W)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
                merge_funcs = None
                global_merging = None
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)
                # Global attention - apply merging if enabled
                if self.do_global_merging:
                    if self.merging is None:
                        global_merging = i  # Pass block number even when merging disabled
                        merge_funcs = None
                    elif self.do_global_merging and i >= self.merging:
                        global_merging = i
                        merge_funcs = merging_functions.get(i, None)
                    else:
                        global_merging = i  # Pass block number even when merging disabled
                        merge_funcs = None
                else:
                    global_merging = None
            
            hidden = blk(hidden, xpos=pos, global_merging=global_merging, merge_funcs=merge_funcs)
            
            # Additional cleanup every few blocks
            if i % 6 == 0:  # Every 6 blocks
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1).detach().cpu())

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def _precalculate_merging_functions(self, hidden, N, H, W):
        """Pre-calculate merging functions once for the first global attention block."""
        B = hidden.shape[0] // N
        hw = hidden.shape[1]
        
        # Calculate grid dimensions
        w, h = H // 14, W // 14
        tokens_per_frame = w * h
        num_frames = N
        
        # Check if we have the expected token structure
        if (tokens_per_frame * num_frames + 5*num_frames)/N != hidden.shape[1]:
            return {}
        
        # Only calculate merging functions once for the first global attention block
        first_global_block = None
        for i in range(1, len(self.decoder), 2):  # Only odd-numbered blocks (global attention)
            if i >= self.merging:
                first_global_block = i
                break
        
        if first_global_block is None:
            return {}
        
        # Reshape for global attention
        hidden_global = hidden.reshape(B, N*hw, -1)
        
        # Import merging functions
        from ..merging.merge import token_merge_pi3
        
        # Calculate merging parameters
        generator = torch.Generator(device=hidden.device)
        generator.manual_seed(33)
        r = int(hidden_global.shape[1] * self.merge_ratio)
        print(f"We reduce the token count from {hidden_global.shape[1]} to {hidden_global.shape[1] - r} for global attention.")
        
        try:
            m_a, u_a = token_merge_pi3(
                hidden_global, w, h, 2, 2, r, False, generator, enable_protection=True
            )
            
            # Create merging functions for all global attention blocks
            merging_functions = {}
            for i in range(1, len(self.decoder), 2):  # All odd-numbered blocks (global attention)
                if i >= self.merging:
                    merging_functions[i] = (m_a, u_a)
            
            # Force garbage collection after pre-calculation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return merging_functions
            
        except Exception as e:
            return {}
    
    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        # load encoder to cuda if on cpu
        self.encoder = self.encoder.cuda()
        hidden = self.encoder(imgs, is_training=True)
        # free memory
        self.encoder = self.encoder.cpu()

        del imgs
        torch.cuda.empty_cache()

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        
        self.decoder = self.decoder.cuda()

        hidden, pos = self.decode(hidden, N, H, W)
        hidden = hidden.to(pos.device)

        if self.is_distillation:
            hidden = self.project_stud_to_teach(hidden)

        # free memory
        self.decoder = self.decoder.cpu()

        # cleanup cuda cache
        torch.cuda.empty_cache()

        # local points
        point_hidden = self.point_decoder(hidden, xpos=pos)
        ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
        if not self.is_distillation:
            del point_hidden
            torch.cuda.empty_cache()

        xy, z = ret.split([2, 1], dim=-1)
        z = torch.exp(z)
        local_points = torch.cat([xy * z, z], dim=-1)

        # confidence
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
        if not self.is_distillation:
            del conf_hidden
            torch.cuda.empty_cache()

        # camera
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)
        if not self.is_distillation:
            del camera_hidden
            torch.cuda.empty_cache()

        # unproject local points using camera poses
        #points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

            
        return dict(
            #points=points.float(),
            local_points=local_points.float(),
            conf=conf.float(),
            camera_poses=camera_poses.float(),
            hidden=hidden.float()
        )
