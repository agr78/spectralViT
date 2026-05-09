"""
Neural network architectures for Spectral Vision Transformer experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class SpectralViT(nn.Module):
    def __init__(
        self, 
        n_inputs, 
        n_heads=2, 
        embed_dim=16, 
        n_layers=4, 
        use_mode_weights=False,
        # New arguments for exact replication
        use_input_proj=True,    # If False, use raw scalar (d_model=1)
        use_pos_embed=True,     # If False, no positional encoding
        pooling='mean',         # 'mean' or 'flatten'
        use_layer_norm=True,    # If False, no LN in head
        use_sigmoid=False       # If True, apply sigmoid to output
    ):
        super().__init__()
        self.use_mode_weights = use_mode_weights
        self.use_pos_embed = use_pos_embed
        self.pooling = pooling
        self.use_sigmoid = use_sigmoid
        self.use_input_proj = use_input_proj

        # 1. Spectral decay
        ranks = torch.arange(1, n_inputs + 1, dtype=torch.float32)
        self.rank_weights = nn.Parameter(1.0 / ranks)

        # Determine the Transformer's d_model
        # DBS mode effectively sets d_model to 1
        d_model = embed_dim if (use_input_proj or use_mode_weights) else 1

        # 2. Projection Logic
        if self.use_mode_weights:
            self.mode_weights = nn.Parameter(torch.randn(n_inputs, d_model) * 0.02)
        elif self.use_input_proj:
            self.input_proj = nn.Linear(1, d_model)
        else:
            # For DBS: No projection, raw scalars are used
            self.input_proj = nn.Identity()

        # 3. Positional Embedding
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(n_inputs, 1, d_model) * 0.02)

        # 4. Transformer
        # Note: dim_feedforward is embed_dim * 2 to match DBS style
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=embed_dim * 2, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 5. Output Head Logic
        head_in_features = d_model if pooling == 'mean' else (d_model * n_inputs)
        
        if use_layer_norm:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(head_in_features), 
                nn.Linear(head_in_features, 1)
            )
        else:
            self.mlp_head = nn.Linear(head_in_features, 1)

    def forward(self, x, return_logit=False):
        # x: [batch, n_inputs]
        if x.ndim == 1: x = x.unsqueeze(0)
        
        # 1. Weighting
        x = x * self.rank_weights
        
        # 2. Projection to sequence: [seq_len, batch, d_model]
        if self.use_mode_weights:
            x = x.unsqueeze(-1) * self.mode_weights.unsqueeze(0)
            x = x.transpose(0, 1)
        else:
            # If Identity (DBS), this just unsqueezes to (seq, batch, 1)
            x = self.input_proj(x.unsqueeze(-1)).transpose(0, 1)
        
        # 3. Positional Embedding
        if self.use_pos_embed:
            x = x + self.pos_embed
            
        # 4. Transformer
        x = self.transformer(x)
        
        # 5. Pooling / Flattening
        if self.pooling == 'mean':
            x = x.mean(dim=0)
        else:
            # Flatten to [batch, n_inputs * d_model]
            x = x.transpose(0, 1).flatten(1)
            
        # 6. Head
        logit = self.mlp_head(x).squeeze(-1)
        
        # Consistency for single-sample batches
        if logit.ndim == 0: logit = logit.unsqueeze(0)

        if self.use_sigmoid and not return_logit:
            return torch.sigmoid(logit)
        return logit

class SpatialViT(nn.Module):
    """
    Unified Spatial Vision Transformer supporting 2D/3D and multiple pooling modes.
    """
    def __init__(self, size=96, patch_size=12, embed_dim=128, n_heads=4, n_layers=2, 
                 dropout=0.2, is_2d=False, use_cls_token=True, use_layer_norm=True, use_sigmoid=False):
        super().__init__()
        self.patch_size = patch_size
        self.is_2d = is_2d
        self.use_cls_token = use_cls_token
        self.use_sigmoid = use_sigmoid
        
        # 1. Architecture Dimensions
        if is_2d:
            self.n_patches = (size // patch_size) ** 2
            proj_in = patch_size ** 2
        else:
            self.n_patches = (size // patch_size) ** 3
            proj_in = patch_size ** 3

        self.proj = nn.Linear(proj_in, embed_dim)
        
        # 2. Sequence Tokens & Positional Embedding
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            pos_len = self.n_patches + 1
        else:
            pos_len = self.n_patches

        # Standardizing pos_embed shape to (seq_len, 1, embed_dim) for broadcasting
        self.pos_embed = nn.Parameter(torch.randn(pos_len, 1, embed_dim) * 0.02)

        # 3. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=embed_dim * 2, 
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Output Head
        if use_layer_norm:
            self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        else:
            self.head = nn.Linear(embed_dim, 1)

    def forward(self, x, return_logit=False):
        b = x.shape[0]
        p = self.patch_size

        # 1. Unfolding logic (DBS is 2D, Standard is 3D)
        if self.is_2d:
            x = x.unfold(2, p, p).unfold(3, p, p)
        else:
            x = x.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)

        # 2. Projection and Sequence formatting: (Seq, Batch, Dim)
        x = self.proj(x.contiguous().view(b, self.n_patches, -1)).transpose(0, 1)

        # 3. Concatenate CLS Token if required
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(1, b, -1)
            x = torch.cat((cls_tokens, x), dim=0)

        # 4. Add Positional Embedding & Transform
        x = x + self.pos_embed
        x = self.transformer(x)

        # 5. Pooling (DBS uses Mean Pooling, Standard uses CLS Token)
        if self.use_cls_token:
            x_pool = x[0]
        else:
            x_pool = x.mean(dim=0)

        logit = self.head(x_pool).squeeze(-1)

        # Consistency for single-sample batches
        if logit.ndim == 0: logit = logit.unsqueeze(0)

        # 6. Activation (DBS returns sigmoid by default)
        if self.use_sigmoid and not return_logit:
            return torch.sigmoid(logit)
        return logit


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


class AttentionUNet3D(nn.Module):
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
        
        # AttentionBlock3D from your library (matching F_g, F_l, F_int logic)
        self.attn = AttentionBlock3D(
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


class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # 1. No BN here
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, kernel_size=1))
        # 2. No BN here
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, kernel_size=1))
        # 3. BN only here (1 channel output = 2 params: weight & bias)
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1), 
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

class ManualSwin(nn.Module):
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

# ============================================================================
# DBS models - EXACT copies from dbs.ipynb
# ============================================================================

class ClinicalTransformerDBS(nn.Module):
    """EXACT copy from dbs.ipynb."""
    def __init__(self, n_inputs, embed_dim=32, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*2, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc = nn.Linear(embed_dim * n_inputs, 1)

    def forward(self, x, return_logit=False):
        if x.ndim == 1: x = x.unsqueeze(0)
        x = self.embedding(x.unsqueeze(-1)).permute(1,0,2)
        x = self.transformer(x).permute(1,0,2).flatten(1)
        logit = self.fc(x).squeeze()
        if logit.ndim == 0: logit = logit.unsqueeze(0)
        return logit if return_logit else torch.sigmoid(logit)


class SpectralViTDBS(nn.Module):
    """EXACT copy from dbs.ipynb."""
    def __init__(self, n_inputs, n_heads=1, embed_dim=32, n_layers=1):
        super().__init__()
        ranks = torch.arange(1, n_inputs+1, dtype=torch.float32)
        self.rank_weights = nn.Parameter(1.0 / ranks)
        layer = nn.TransformerEncoderLayer(d_model=1, nhead=n_heads, dim_feedforward=embed_dim*2, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc = nn.Linear(n_inputs, 1)

    def forward(self, x, return_logit=False):
        x_weighted = x * self.rank_weights
        x_seq = x_weighted.unsqueeze(-1).transpose(0,1)
        x_trans = self.transformer(x_seq)
        x_trans_flat = x_trans.transpose(0,1).squeeze(-1)
        logit = self.fc(x_trans_flat).squeeze()
        if logit.ndim == 0: logit = logit.unsqueeze(0)
        return logit if return_logit else torch.sigmoid(logit)


class SpatialViTDBS(nn.Module):
    """EXACT copy from dbs.ipynb."""
    def __init__(self, img_size=128, patch_size=16, embed_dim=32, n_heads=1, n_layers=1):
        super().__init__()
        self.patch_size, self.n_patches = patch_size, (img_size // patch_size)**2
        self.proj = nn.Linear(patch_size*patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*2, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, return_logit=False):
        b = x.shape[0]
        x = x.unfold(2,self.patch_size,self.patch_size).unfold(3,self.patch_size,self.patch_size)
        x = self.proj(x.contiguous().view(b, self.n_patches, -1)) + self.pos_embed
        x = self.transformer(x.permute(1,0,2)).permute(1,0,2).mean(dim=1)
        logit = self.fc(x).squeeze()
        if logit.ndim == 0: logit = logit.unsqueeze(0)
        return logit if return_logit else torch.sigmoid(logit)


class ResidualSpectral(nn.Module):
    """EXACT copy from dbs.ipynb."""
    def __init__(self, m_ct, m_res):
        super().__init__()
        self.m_ct, self.m_res = m_ct, m_res
    def forward(self, x_res, x_clin, return_logit=False):
        with torch.no_grad():
            l_clin = self.m_ct(x_clin, return_logit=True)
        l_res = self.m_res(x_res, return_logit=True)
        joint = l_clin + l_res
        return joint if return_logit else torch.sigmoid(joint)


class ResidualSpatial(nn.Module):
    """EXACT copy from dbs.ipynb."""
    def __init__(self, m_ct, m_res):
        super().__init__()
        self.m_ct, self.m_res = m_ct, m_res
    def forward(self, x_res, x_clin, return_logit=False):
        with torch.no_grad():
            l_clin = self.m_ct(x_clin, return_logit=True)
        l_res = self.m_res(x_res, return_logit=True)
        joint = l_clin + l_res
        return joint if return_logit else torch.sigmoid(joint)


class SpatialAttention(nn.Module):
    """EXACT copy from dbs.ipynb."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(att))
        return x * (1 + att)


class FastUNet(nn.Module):
    """EXACT copy from dbs.ipynb."""
    def __init__(self, in_channels=1, out_classes=1):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True))
        self.enc1 = conv_block(in_channels, 16)
        self.enc2 = conv_block(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(32, 64)
        self.spatial_att = SpatialAttention()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_classes))
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        bn = self.bottleneck(self.pool(x2))
        bn = self.spatial_att(bn)
        pooled = self.gmp(bn).view(x.size(0), -1)
        return self.fc(pooled).squeeze(-1)


class PCAClinMLP(nn.Module):
    """EXACT copy from dbs.ipynb."""
    def __init__(self, n_inputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
    def forward(self, x, return_logit=False):
        logit = self.net(x).squeeze()
        if logit.ndim == 0: logit = logit.unsqueeze(0)
        return logit if return_logit else torch.sigmoid(logit)


class PassThrough(nn.Module):
    """EXACT copy from dbs.ipynb."""
    def forward(self, x, **kwargs):
        return x


class FocalLoss(nn.Module):
    """EXACT copy from dbs.ipynb."""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets, weight=None):
        bce = F.binary_cross_entropy(inputs, targets, weight=weight, reduction='none')
        return (self.alpha * (1 - torch.exp(-bce))**self.gamma * bce).mean()