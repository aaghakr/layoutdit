import torch
from torch import nn
from cgbdm.vit import ViT
from cgbdm.module import LayoutModule, MLP, Q_former
from typing import Optional
from cgbdm.swin import Swin


class LayoutModel(nn.Module):
    def __init__(self,
                 num_layers: int = 4,
                 dim_seq: int = 8,
                 dim_model: int = 512,
                 n_head: int = 8,
                 dim_feedforward: int = 1024,
                 diffusion_steps: int = 1000,
                 max_elem: int = 16,
                 v_encoder: str = 'vit',
                 spatial_guidance: int = 0,
                 text_control: bool = False,
                 text_conditioning_mode: str = 'token',
                 text_guidance_scale: float = 1.0,
                 device: Optional[str] = None
                 ):
        super(LayoutModel, self).__init__()
        self.spatial_guidance = spatial_guidance
        self.text_control = text_control
        if text_conditioning_mode not in {'token', 'pooled'}:
            raise ValueError(f"Unknown text conditioning mode: {text_conditioning_mode}")
        self.text_conditioning_mode = text_conditioning_mode
        self.text_guidance_scale = float(text_guidance_scale)

        common_params = {
            'dim_seq': dim_seq,
            'dim_model': dim_model,
            'nhead': n_head,
            'dim_feedforward': dim_feedforward,
            'diffusion_steps': diffusion_steps,
            'max_elem': max_elem,
            'device': device
        }

        # Determine number of channels based on spatial guidance
        if spatial_guidance == 0:  # Saliency only
            num_channels = 4  # RGB + Saliency
        elif spatial_guidance == 1:  # Intent only
            num_channels = 4  # RGB + Intent
        elif spatial_guidance == 2:  # Both
            num_channels = 5  # RGB + Saliency + Intent
        elif spatial_guidance == 3:  # Saliency pixels + intent boxes
            num_channels = 4
        else:
            num_channels = 4  # Default fallback

        if v_encoder == 'vit':
            self.img_encoder = ViT(image_size=[384,256],patch_size=32,channels=num_channels,
                                   dim=512,depth=6,heads=8,mlp_dim=2048,dropout=0.1,emb_dropout=0.1).to(device)
        elif v_encoder == 'swin':
            self.img_encoder = Swin(image_size=[384,256],patch_size=32,channels=num_channels,
                                   dim=512,depth=6,heads=8,mlp_dim=2048,dropout=0.1,emb_dropout=0.1).to(device)

        self.layout_encoder = LayoutModule(
            num_layers=num_layers // 2,
            if_encoder=True,
            **common_params
        )
        self.layout_decoder = LayoutModule(
            num_layers=num_layers,
            if_encoder=False,
            **common_params
        )
        self.cgbwp = Q_former(in_dim=dim_model, num_tokens=1).to(device)
        self.salbox_encoder = MLP(input_dim=4, hidden_dim=dim_model, output_dim=dim_model, num_layers=3).to(device)
        self.intent_encoder = MLP(input_dim=4, hidden_dim=dim_model, output_dim=dim_model, num_layers=3).to(device)

        # Text encoder for text control
        self.text_control = getattr(self, 'text_control', False)
        if self.text_control:
            from transformers import BertModel
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased').to(device)
            if getattr(self, 'freeze_text_encoder', True):
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            self.text_projection = nn.Linear(768, dim_model).to(device)

            # Text-spatial encoder: encodes (class_one_hot + position_box) into
            # embeddings that can replace intent_encode when user specifies positions.
            # Input dim = dim_seq = num_class + 4 (same layout representation).
            self.text_spatial_encoder = MLP(
                input_dim=dim_seq,
                hidden_dim=dim_model,
                output_dim=dim_model,
                num_layers=3,
            ).to(device)

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, layout, image, sal_box, timestep, intent_box=None,
                text_features=None, text_spatial_boxes=None,
                text_spatial_mask=None, use_text_spatial=None):
        img_encode = self.img_encoder(image)
        salbox_encode = self.salbox_encoder(sal_box)

        # --- Spatial conditioning (intent vs text-spatial) ---
        intent_encode = None
        spatial_key_padding_mask = None
        spatial_condition_active = None

        if self.spatial_guidance in (1, 2, 3) and intent_box is not None:
            intent_encode = self.intent_encoder(intent_box)
            # Padded placement boxes use [-1, -1, -1, -1], which decodes to
            # zero width/height. Keep their MLP embeddings out of attention.
            spatial_key_padding_mask = (
                (intent_box[..., 2] <= -1.0 + 1e-6)
                | (intent_box[..., 3] <= -1.0 + 1e-6)
            )

        # When text_spatial_boxes are provided, override intent for relevant items
        if (self.text_control and text_spatial_boxes is not None
                and use_text_spatial is not None and use_text_spatial.any()):
            ts_encode = self.text_spatial_encoder(text_spatial_boxes)

            if intent_encode is not None and not use_text_spatial.all():
                # Mixed batch: per-item selection
                sel = use_text_spatial[:, None, None].expand_as(intent_encode)
                intent_encode = torch.where(sel, ts_encode, intent_encode)
                if text_spatial_mask is not None:
                    spatial_key_padding_mask = torch.where(
                        use_text_spatial[:, None].expand_as(text_spatial_mask),
                        text_spatial_mask,
                        spatial_key_padding_mask,
                    )
            elif intent_encode is not None:
                intent_encode = ts_encode
                spatial_key_padding_mask = text_spatial_mask
            else:
                # Saliency-only text batches have no intent fallback. Mixed
                # batches therefore contain rows with every spatial key
                # masked, which makes MultiheadAttention return NaNs. Keep one
                # harmless key visible for those rows, then gate the complete
                # spatial update off for them.
                intent_encode = ts_encode
                spatial_key_padding_mask = text_spatial_mask.clone()
                inactive = ~use_text_spatial
                spatial_key_padding_mask[inactive, 0] = False
                spatial_condition_active = use_text_spatial

        # --- Text features (BERT token-level cross-attention) ---
        text_encode = None
        text_key_padding_mask = None
        if self.text_control and text_features is not None:
            with torch.no_grad():
                text_outputs = self.text_encoder(
                    input_ids=text_features['input_ids'],
                    attention_mask=text_features['attention_mask'],
                )
                token_hidden = text_outputs.last_hidden_state
            if self.text_conditioning_mode == 'pooled':
                token_hidden = token_hidden[:, :1, :]
                text_key_padding_mask = torch.zeros(
                    token_hidden.shape[:2], dtype=torch.bool, device=token_hidden.device
                )
            else:
                text_key_padding_mask = (text_features['attention_mask'] == 0)
            text_encode = self.text_projection(token_hidden)
            text_encode = text_encode * self.text_guidance_scale

        layout_encode = self.layout_encoder(layout, None, None, None, None, None, timestep)
        cgb_w = self.cgbwp(img_encode)
        output = self.layout_decoder(
            layout_encode, img_encode, cgb_w, salbox_encode,
            intent_encode, text_encode, timestep,
            text_key_padding_mask=text_key_padding_mask,
            spatial_key_padding_mask=spatial_key_padding_mask,
            spatial_condition_active=spatial_condition_active,
        )

        return output
