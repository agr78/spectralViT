"""
Neural network architectures for Spectral Vision Transformer experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class SpectralViT(nn.Module):
    """
    Spectral Vision Transformer that operates on PCA-transformed features.
    Supports both shared linear projection and mode-specific weighting.
    """
    def __init__(self, n_inputs, n_heads=2, embed_dim=16, n_layers=4, use_mode_weights=False):
        super().__init__()
        self.use_mode_weights = use_mode_weights
        # Spectral decay coefficients (1/f weighting)
        ranks = torch.arange(1, n_inputs + 1, dtype=torch.float32)
        self.rank_weights = nn.Parameter(1.0 / ranks)
        if self.use_mode_weights:
            # Component-wise projection
            self.mode_weights = nn.Parameter(torch.randn(n_inputs, embed_dim) * 0.02)
        else:
            # Shared projection
            self.input_proj = nn.Linear(1, embed_dim)
        # Positional embeddings to maintain identity of each mode
        self.pos_embed = nn.Parameter(torch.randn(n_inputs, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=embed_dim * 2, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), 
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        # x shape: [batch, n_inputs]
        # Apply rank weighting
        x = x * self.rank_weights
        # Projection
        if self.use_mode_weights:
            # Result shape: [batch, n_inputs, embed_dim]
            x = x.unsqueeze(-1) * self.mode_weights.unsqueeze(0)
            x = x.transpose(0, 1) # [seq_len, batch, embed_dim]
        else:
            # Result shape: [batch, n_inputs, embed_dim]
            x = self.input_proj(x.unsqueeze(-1))
            x = x.transpose(0, 1) # [seq_len, batch, embed_dim]
        # 3. Add Positional Encoding & Transformer
        x = x + self.pos_embed
        x = self.transformer(x)
        # 4. Global Average Pooling & Head
        x = x.mean(dim=0)
        return self.mlp_head(x).squeeze(-1)

class SpatialViT(nn.Module):
    """
    Spatial Vision Transformer that operates on 3D image patches.
    """
    def __init__(self, vol_size=96, patch_size=12, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (vol_size // patch_size) ** 3
        self.proj = nn.Linear(patch_size ** 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(self.n_patches + 1, 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=embed_dim * 2, 
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

    def forward(self, x):
        b = x.shape[0]
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p).contiguous().view(b, self.n_patches, -1)
        x = self.proj(x).transpose(0, 1)
        cls_tokens = self.cls_token.expand(1, b, -1)
        x = torch.cat((cls_tokens, x), dim=0)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.mlp_head(x[0]).squeeze(-1)


class SpatialViTMatched(nn.Module):
    """
    Spatial Vision Transformer with matched parameter count to SpectralViT.
    Uses smaller embedding dimension.
    """
    def __init__(self, vol_size=96, patch_size=12, embed_dim=16, n_heads=2, n_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (vol_size // patch_size) ** 3
        self.proj = nn.Linear(patch_size ** 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(self.n_patches + 1, 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

    def forward(self, x):
        b = x.shape[0]
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p).contiguous().view(b, self.n_patches, -1)
        x = self.proj(x).transpose(0, 1)
        cls_tokens = self.cls_token.expand(1, b, -1)
        x = torch.cat((cls_tokens, x), dim=0)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.mlp_head(x[0]).squeeze(-1)


class AttentionUNet(nn.Module):
    """
    Identical refactor of CompactAttnUNet.
    3-layer encoder using strided convolutions and latent self-attention.
    """
    def __init__(self, in_channels=1, base_channels=7):
        super().__init__()
        # Encoder matching CompactAttnUNet logic
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, stride=2) 
        self.conv2 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1, stride=2) 
        self.conv3 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1, stride=2) 
        
        # AttentionBlock
        self.attn = AttentionBlock(
            F_g=base_channels*4, 
            F_l=base_channels*4, 
            F_int=base_channels*2
        )
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(base_channels*4, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        
        # Self-attention on the bottleneck/latent space
        g = self.attn(g=x3, x=x3)
        
        out = self.pool(g).view(g.size(0), -1)
        return self.classifier(out).squeeze(-1)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# Shifted window (Swin) transformer
class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=6, shift_size=0):
        super().__init__()
        self.dim, self.input_resolution = dim, input_resolution
        self.window_size, self.shift_size = window_size, shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*2, dropout=0.1)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Window Partitioning
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)

        # Attention
        attn_windows = self.attn(windows.transpose(0, 1)).transpose(0, 1)

        # Merge Windows
        x = attn_windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        return shortcut + self.norm2(x.view(B, L, C))

class SwinTransformer(nn.Module):
    def __init__(self, img_size=96, patch_size=8, embed_dim=32, window_size=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        res = img_size // patch_size
        self.layers = nn.ModuleList([
            SwinBlock(embed_dim, (res, res), num_heads=4, window_size=window_size, shift_size=0),
            SwinBlock(embed_dim, (res, res), num_heads=4, window_size=window_size, shift_size=window_size // 2)
        ])
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # Collapse 3D to 2D for Swin blocks
        if x.dim() == 5: x = x.mean(dim=2) 
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1) 
        return self.head(x).squeeze(-1)


class ClinicalTransformer(nn.Module):
    """
    Transformer for clinical/tabular data.
    """
    def __init__(self, n_features, embed_dim=32, n_heads=2, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(n_features, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x: (batch, n_features)
        x = self.embed(x.unsqueeze(1))  # (batch, 1, embed_dim)
        x = x + self.pos_embed
        x = x.transpose(0, 1)  # (1, batch, embed_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch, embed_dim)
        return self.head(x).squeeze(-1)
