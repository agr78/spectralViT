import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralViT(nn.Module):
    """
    Fully Unified Spectral ViT.
    Works for:
    1. Simulation script (needs learnable_rank_weights=False, use_layer_norm=False)
    2. Medical script (needs use_input_proj, use_pos_embed, use_mode_weights)
    """
    def __init__(
        self, 
        n_inputs, 
        n_heads=2, 
        embed_dim=16, 
        n_layers=4, 
        use_mode_weights=False,
        use_input_proj=True,    # Restored
        use_pos_embed=True,     # Restored
        pooling='mean',         
        use_layer_norm=True,    
        learnable_rank_weights=True,
        use_sigmoid=False       
    ):
        super().__init__()
        self.use_mode_weights = use_mode_weights
        self.use_pos_embed = use_pos_embed
        self.pooling = pooling
        self.use_sigmoid = use_sigmoid
        self.use_input_proj = use_input_proj

        # 1. Spectral decay (1/f) logic
        ranks = torch.arange(1, n_inputs + 1, dtype=torch.float32)
        if learnable_rank_weights:
            self.rank_weights = nn.Parameter(1.0 / ranks)
        else:
            self.register_buffer('rank_weights', 1.0 / ranks)

        # 2. Input Projection Logic
        # DBS/IXI behavior uses d_model=1 or d_model=embed_dim
        self.d_model = embed_dim if (use_input_proj or use_mode_weights) else 1

        if self.use_mode_weights:
            self.mode_weights = nn.Parameter(torch.randn(n_inputs, self.d_model) * 0.02)
        elif self.use_input_proj:
            self.input_proj = nn.Linear(1, self.d_model)
        else:
            self.input_proj = nn.Identity()

        # 3. Positional Embedding Logic
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(n_inputs, 1, self.d_model) * 0.02)

        # 4. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=n_heads, 
            dim_feedforward=embed_dim * 2, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 5. Head Logic
        head_in = self.d_model if pooling == 'mean' else (self.d_model * n_inputs)
        
        if use_layer_norm:
            self.mlp_head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, 1))
        else:
            self.mlp_head = nn.Linear(head_in, 1)

    def forward(self, x, return_logit=False):
        if x.ndim == 1: x = x.unsqueeze(0)
        
        # Apply 1/f weighting
        x = x * self.rank_weights
        
        # Apply Projection
        if self.use_mode_weights:
            x = x.unsqueeze(-1) * self.mode_weights.unsqueeze(0)
            x = x.transpose(0, 1)
        else:
            # Result: [seq_len, batch, d_model]
            x = self.input_proj(x.unsqueeze(-1)).transpose(0, 1)
        
        # Add Position
        if self.use_pos_embed:
            x = x + self.pos_embed
            
        x = self.transformer(x)
        
        # Pooling
        if self.pooling == 'mean':
            x = x.mean(dim=0)
        else:
            x = x.transpose(0, 1).flatten(1)
            
        logit = self.mlp_head(x).squeeze(-1)
        if logit.ndim == 0: logit = logit.unsqueeze(0)

        if self.use_sigmoid and not return_logit:
            return torch.sigmoid(logit)
        return logit

class SpatialViT(nn.Module):
    """
    Unified Spatial ViT.
    DBS behavior: is_2d=True, use_cls_token=False (uses mean)
    IXI behavior: is_2d=False, use_cls_token=True, vol_size=96
    """
    def __init__(self, size=96, vol_size=None, patch_size=12, embed_dim=128, n_heads=4, n_layers=2, 
                 dropout=0.2, is_2d=False, use_cls_token=True, use_layer_norm=True, use_sigmoid=False):
        super().__init__()
        # Compatibility for 'size' (DBS) vs 'vol_size' (IXI)
        actual_size = vol_size if vol_size is not None else size
        self.patch_size = patch_size
        self.is_2d = is_2d
        self.use_cls_token = use_cls_token
        self.use_sigmoid = use_sigmoid
        
        if is_2d:
            self.n_patches = (actual_size // patch_size) ** 2
            proj_in = patch_size ** 2
        else:
            self.n_patches = (actual_size // patch_size) ** 3
            proj_in = patch_size ** 3

        self.proj = nn.Linear(proj_in, embed_dim)
        
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            pos_len = self.n_patches + 1
        else:
            pos_len = self.n_patches

        self.pos_embed = nn.Parameter(torch.randn(pos_len, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if use_layer_norm:
            self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        else:
            self.head = nn.Linear(embed_dim, 1)

    def forward(self, x, return_logit=False):
        b = x.shape[0]
        p = self.patch_size
        if self.is_2d:
            x = x.unfold(2, p, p).unfold(3, p, p)
        else:
            x = x.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)

        x = self.proj(x.contiguous().view(b, self.n_patches, -1)).transpose(0, 1)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(1, b, -1)
            x = torch.cat((cls_tokens, x), dim=0)

        x = x + self.pos_embed
        x = self.transformer(x)

        x_pool = x[0] if self.use_cls_token else x.mean(dim=0)
        logit = self.head(x_pool).squeeze(-1)
        
        if logit.ndim == 0: logit = logit.unsqueeze(0)
        if self.use_sigmoid and not return_logit:
            return torch.sigmoid(logit)
        return logit


class ClinicalTransformer(nn.Module):
    """DBS Logic: Each feature = One token."""
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
    

class ResidualSpectral(nn.Module):
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
    def __init__(self, m_ct, m_res):
        super().__init__()
        self.m_ct, self.m_res = m_ct, m_res
    def forward(self, x_res, x_clin, return_logit=False):
        with torch.no_grad():
            l_clin = self.m_ct(x_clin, return_logit=True)
        l_res = self.m_res(x_res, return_logit=True)
        joint = l_clin + l_res
        return joint if return_logit else torch.sigmoid(joint)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets, weight=None):
        bce = F.binary_cross_entropy(inputs, targets, weight=weight, reduction='none')
        return (self.alpha * (1 - torch.exp(-bce))**self.gamma * bce).mean()

class PassThrough(nn.Module):
    def forward(self, x, **kwargs): return x




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


