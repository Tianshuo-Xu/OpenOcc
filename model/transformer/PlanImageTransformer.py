from mmengine.registry import MODELS
from mmengine.model import BaseModule
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from einops import rearrange


@MODELS.register_module()
class CustomAutoRegressiveModel(BaseModule):
    def __init__(
            self, 
            base_embed_dim,
            occupancy_embed_dim=128, 
            img_embed_dim=24, 
            pose_embed_dim=128,
            n_layer=2,
            n_head=4,
        ):
        super().__init__()
        self.occupancy_conv = nn.Conv3d(occupancy_embed_dim, base_embed_dim, kernel_size=(1, 1, 1))
        self.img_conv = nn.Conv3d(img_embed_dim, base_embed_dim, kernel_size=(1, 1, 1))
        self.pose_embed = nn.Linear(pose_embed_dim, base_embed_dim)

        config = GPT2Config(
            n_embd=base_embed_dim,      # Embedding dimension
            n_layer=n_layer,       # Number of layers
            n_head=n_head       # Number of attention heads
        )
        self.transformer = GPT2Model(config)

        self.occupancy_deconv = nn.Conv3d(base_embed_dim, occupancy_embed_dim, kernel_size=(1, 1, 1))
        self.img_deconv = nn.Conv3d(base_embed_dim, img_embed_dim, kernel_size=(1, 1, 1))
        self.pose_deembed = nn.Linear(base_embed_dim, pose_embed_dim)

    def forward(self, occupancy_token, pose_token, img_token):
        bs, f, c, h, w = occupancy_token.shape

        occupancy_token = rearrange(occupancy_token, 'b f c h w -> b c f h w')
        occupancy_flat = self.occupancy_conv(occupancy_token)
        occupancy_flat = rearrange(occupancy_flat, 'b c f h w -> b (f h w) c')

        img_token = rearrange(img_token, 'b f c h w -> b c f h w')
        img_flat = self.img_conv(img_token)
        img_flat = rearrange(img_flat, 'b c f h w -> b (f h w) c')

        pose_flat = self.pose_embed(pose_token)

        print(occupancy_flat.shape, pose_flat.shape, img_flat.shape)
        # Concatenate along the embedding dimension
        transformer_input = torch.cat((occupancy_flat, pose_flat, img_flat), dim=1)
        
        # Pass through transformer
        transformer_output = self.transformer(inputs_embeds=transformer_input).last_hidden_state
        
        # Reshape and pass through deconvolution or de-embedding layers
        occupancy_output = self.occupancy_deconv(transformer_output[:, :, :occupancy_embed_dim].view(batch_size, timesteps, -1, H, W))
        pose_output = self.pose_deembed(transformer_output[:, :, occupancy_embed_dim:occupancy_embed_dim + pose_embed_dim])
        img_output = self.img_deconv(transformer_output[:, :, occupancy_embed_dim + pose_embed_dim:].view(batch_size, timesteps, -1, H, W))
        
        return occupancy_output, pose_output, img_output


if __name__ == '__main__':
    # Hyperparameters for embedding dimensions
    base_embed_dim = 128
    occupancy_embed_dim = 128
    img_embed_dim = 24

    # Instantiate the custom model
    model = CustomAutoRegressiveModel(base_embed_dim, occupancy_embed_dim, img_embed_dim)

    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} parameters")

    occ_tokens = torch.ones((1, 15, 128, 50, 50), dtype=torch.float32)
    pose_tokens = torch.ones((1, 15, 128))
    img_tokens = torch.ones((1, 15, 24, 36, 64), dtype=torch.float32)
    o, p, i = model(occ_tokens, pose_tokens, img_tokens)
    print(o.shape, p.shape, i.shape)
