import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, eye
from sklearn.decomposition import PCA
import numpy as np

class SpectralViT(nn.Module):
    def __init__(
        self, 
        n_inputs, 
        n_heads=2, 
        embed_dim=16, 
        n_layers=4, 
        patch_size=1,          
        sampling_indices=None,  
        use_rank_weights=True,  
        use_mode_weights=False,
        use_input_proj=True,    
        use_pos_embed=True,     
        pooling='mean',         
        use_layer_norm=True,    
        learnable_rank_weights=True,
        use_sigmoid=False       
    ):
        super().__init__()
        self.patch_size = patch_size
        self.use_rank_weights = use_rank_weights
        self.use_mode_weights = use_mode_weights
        self.use_pos_embed = use_pos_embed
        self.pooling = pooling
        self.use_sigmoid = use_sigmoid
        self.use_input_proj = use_input_proj

        if sampling_indices is not None:
            self.register_buffer('sampling_indices', sampling_indices)
        else:
            self.sampling_indices = None

        # 1. Rank weights logic (1/f)
        if use_rank_weights:
            ranks = torch.arange(1, n_inputs + 1, dtype=torch.float32)
            if learnable_rank_weights:
                self.rank_weights = nn.Parameter(1.0 / ranks)
            else:
                self.register_buffer('rank_weights', 1.0 / ranks)

        # 2. Projection Logic
        self.d_model = embed_dim if (use_input_proj or use_mode_weights) else 1
        
        if self.use_mode_weights:
            self.mode_weights = nn.Parameter(torch.randn(n_inputs, self.d_model) * 0.02)
        elif self.use_input_proj:
            self.input_proj = nn.Linear(patch_size, self.d_model)
        else:
            self.input_proj = nn.Identity()

        # 3. Positional Embedding - Force (Seq, 1, Dim) for correct broadcasting
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(n_inputs, 1, self.d_model) * 0.02)

        # 4. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads, 
            dim_feedforward=embed_dim * 2, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 5. Head
        head_in = self.d_model if pooling == 'mean' else (self.d_model * n_inputs)
        if use_layer_norm:
            self.mlp_head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, 1))
        else:
            self.mlp_head = nn.Linear(head_in, 1)

    def forward(self, x, return_logit=False):
        if x.ndim == 1: x = x.unsqueeze(0)
        b = x.shape[0]

        # A. Optional Sampling
        if self.sampling_indices is not None:
            x = x.view(b, -1)[:, self.sampling_indices]

        # B. Rank Weights
        if self.use_rank_weights:
            # Slice rank_weights in case input x is shorter than n_inputs
            x = x * self.rank_weights[:x.size(1)]
        
        # C. Projection Branching
        if self.use_mode_weights:
            # [Batch, Seq] -> [Batch, Seq, 1]
            x = x.unsqueeze(-1) 
            # [Batch, Seq, 1] * [1, Seq, Dim] -> [Batch, Seq, Dim]
            x = x * self.mode_weights[:x.size(1), :].unsqueeze(0)
            # -> [Seq, Batch, Dim]
            x = x.transpose(0, 1)
        else:
            # [Batch, Seq] -> [Batch, Seq, Patch]
            x = x.view(b, -1, self.patch_size)
            # [Batch, Seq, Dim] -> [Seq, Batch, Dim]
            x = self.input_proj(x).transpose(0, 1)
        
        # D. Position Addition (Safely Broadcasted)
        if self.use_pos_embed:
            # Ensure pos_embed matches sequence length and broadcasts over batch (dim 1)
            x = x + self.pos_embed[:x.size(0), :, :]
            
        x = self.transformer(x)
        
        # E. Pooling
        if self.pooling == 'mean':
            x = x.mean(dim=0)
        else:
            # Result: [Batch, Seq * Dim]
            x = x.transpose(0, 1).flatten(1)
            
        logit = self.mlp_head(x).squeeze(-1)
        if logit.ndim == 0: logit = logit.unsqueeze(0)
        if self.use_sigmoid and not return_logit:
            return torch.sigmoid(logit)
        return logit
    
class SpatialViT(nn.Module):
    def __init__(self, size=96, vol_size=None, patch_size=12, embed_dim=128, n_heads=4, n_layers=2, 
                 dropout=0.2, is_2d=False, use_cls_token=True, use_layer_norm=True, use_sigmoid=False):
        super().__init__()
        actual_size = vol_size if vol_size is not None else size
        self.patch_size = patch_size
        self.is_2d = is_2d
        self.use_cls_token = use_cls_token
        self.use_sigmoid = use_sigmoid
        
        # Calculate patches based on init size
        if is_2d:
            self.n_patches = (actual_size // patch_size) ** 2
            proj_in = patch_size ** 2
        else:
            self.n_patches = (actual_size // patch_size) ** 3
            proj_in = patch_size ** 3

        self.proj = nn.Linear(proj_in, embed_dim)
        
        # Positional Embedding (Seq, 1, Dim)
        pos_len = self.n_patches + 1 if use_cls_token else self.n_patches
        self.pos_embed = nn.Parameter(torch.randn(pos_len, 1, embed_dim) * 0.02)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

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
        
        # 1. Dynamic Unfolding
        if self.is_2d:
            x = x.unfold(2, p, p).unfold(3, p, p)
        else:
            x = x.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)

        # 2. View using -1 to handle any input volume size dynamically
        # Shape: [Batch, n_patches, proj_in]
        x = x.contiguous().view(b, -1, self.proj.in_features)
        x = self.proj(x).transpose(0, 1) # [Seq, Batch, Dim]

        # 3. CLS Token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(1, b, -1)
            x = torch.cat((cls_tokens, x), dim=0)

        # 4. DYNAMIC POSITION INTERPOLATION (The Fix)
        if x.size(0) != self.pos_embed.size(0):
            # [Old_Seq, 1, Dim] -> [1, Dim, Old_Seq]
            pe = self.pos_embed.permute(1, 2, 0)
            # Interpolate to New_Seq
            pe = F.interpolate(pe, size=x.size(0), mode='linear', align_corners=False)
            # -> [New_Seq, 1, Dim]
            x = x + pe.permute(2, 0, 1)
        else:
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


class AttentionBlock(nn.Module):
    def __init__(self, F_g=None, F_l=None, F_int=None, is_spatial_2d=False):
        super(AttentionBlock, self).__init__()
        self.is_spatial_2d = is_spatial_2d
        
        if not is_spatial_2d:
            # ORIGINAL 3D GATE
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
        else:
            # EXACT FASTUNET SPATIAL ATTENTION
            self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
            self.sigmoid = nn.Sigmoid()

    def forward(self, g, x=None):
        if self.is_spatial_2d:
            # Logic: x * (1 + att)
            avg_out = torch.mean(g, dim=1, keepdim=True)
            max_out, _ = torch.max(g, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.sigmoid(self.conv(att))
            return g * (1 + att)
        else:
            # Logic: x * psi
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=7, fast_mode=False, use_sigmoid=False):
        super(AttentionUNet, self).__init__()
        self.fast_mode = fast_mode
        self.use_sigmoid = use_sigmoid

        if not fast_mode:
            # --- ORIGINAL 3D MODEL ---
            self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, stride=2) 
            self.conv2 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1, stride=2) 
            self.conv3 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1, stride=2) 
            self.attn = AttentionBlock(base_channels*4, base_channels*4, base_channels*2, is_spatial_2d=False)
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.classifier = nn.Linear(base_channels*4, 1)
        else:
            # --- EXACT FASTUNET REPLICA ---
            def conv_block(in_c, out_c):
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                )
            self.enc1 = conv_block(in_channels, base_channels)    # 16
            self.enc2 = conv_block(base_channels, base_channels*2) # 32
            self.pool2d = nn.MaxPool2d(2)
            self.bottleneck = conv_block(base_channels*2, base_channels*4) # 64
            
            self.attn = AttentionBlock(is_spatial_2d=True)
            self.gmp = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(base_channels*4, base_channels*2), 
                nn.ReLU(inplace=True),
                nn.Linear(base_channels*2, 1)              
            )
            nn.init.zeros_(self.fc[-1].weight)
            nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x, return_logit=False):
        if self.fast_mode:
            # Replicating the 2-pool logic of FastUNet
            if x.dim() == 5: x = x.mean(dim=2)
            x1 = self.enc1(x)
            x2 = self.enc2(self.pool2d(x1))
            bn = self.bottleneck(self.pool2d(x2))
            bn = self.attn(bn)
            pooled = self.gmp(bn).view(x.size(0), -1)
            logit = self.fc(pooled).squeeze(-1)
        else:
            x1 = F.relu(self.conv1(x))
            x2 = F.relu(self.conv2(x1))
            x3 = F.relu(self.conv3(x2))
            g = self.attn(g=x3, x=x3)
            out = self.pool(g).view(g.size(0), -1)
            logit = self.classifier(out).squeeze(-1)

        if logit.ndim == 0: logit = logit.unsqueeze(0)
        if self.use_sigmoid and not return_logit:
            return torch.sigmoid(logit)
        return logit

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
    
def pca_tokenize(img_size, X_train, img_test, k):
    n_comp = min(k, X_train.shape[0])
    pca = PCA(n_components=n_comp).fit(X_train)
    tokens = pca.transform(img_test.flatten().reshape(1, -1))
    img_recon = pca.inverse_transform(tokens).reshape(img_size, img_size)
    return tokens.flatten(), img_recon

def fourier_tokenize(img_size, img, k):
    kspace = fftshift(fft2(img))
    kspace_flat = kspace.flatten()
    idx = np.argsort(np.abs(kspace_flat))[::-1]
    kspace_recon = np.zeros_like(kspace_flat, dtype=complex)
    kspace_recon[idx[:k]] = kspace_flat[idx[:k]]
    img_recon = np.real(ifft2(ifftshift(kspace_recon.reshape(img_size, img_size))))
    # Magnitude tokens provide translation invariance
    return np.abs(kspace_flat[idx[:k]]), img_recon

def laplacian_tokenize(img_size, img_clean, img_noisy, k, threshold=0.03, topo_labels=None):
    n_pixels = img_size * img_size
    img_flat = img_clean.flatten()
    rows, cols, vals = [], [], []
    
    for i in range(n_pixels):
        r, c = divmod(i, img_size)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < img_size and 0 <= nc < img_size:
                j = nr * img_size + nc
                if topo_labels is not None:
                    weight = 1.0 if topo_labels.flat[i] == topo_labels.flat[j] else 0.0001
                else:
                    weight = np.exp(-np.abs(img_flat[i] - img_flat[j])**2 / (2 * threshold**2))
                rows.append(i); cols.append(j); vals.append(weight)
    
    W = csr_matrix((vals, (rows, cols)), shape=(n_pixels, n_pixels))
    d = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = csr_matrix((np.power(d, -0.5, where=d!=0), (range(n_pixels), range(n_pixels))))
    L = eye(n_pixels) - D_inv_sqrt @ W @ D_inv_sqrt
    _, evecs = eigsh(L, k=k, which='SM')
    tokens = evecs.T @ img_noisy.flatten()
    img_recon = (evecs @ tokens).reshape(img_size, img_size)
    return tokens, img_recon


class LogisticRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__(); self.linear = nn.Linear(in_features, 1)
    def forward(self, x): return self.linear(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        A simple 2-layer Multi-Layer Perceptron (MLP).
        
        Args:
            input_dim (int): The number of input features (formerly N_COMP).
            hidden_dim (int): The number of hidden units (formerly MLP_HIDDEN_DIM).
        """
        super(MultiLayerPerceptron, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MLP.
        """
        return self.network(x)
    
class MultiClassSpectralViT(nn.Module):
    def __init__(self, n_inputs, chunk_size=4, num_classes=10, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_tokens = n_inputs // chunk_size
        self.proj = nn.Linear(chunk_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, self.n_tokens, self.chunk_size)
        x = self.proj(x) + self.pos_embed
        x = x.permute(1, 0, 2) # (Seq, Batch, Dim) for legacy
        x = self.transformer(x)
        return self.classifier(x.mean(dim=0))

class MultiClassSpatialViT(nn.Module):
    def __init__(self, num_classes=10, size=28, patch_size=4, in_channels=1, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p).contiguous().view(b, self.n_patches, -1) 
        x = self.proj(x) + self.pos_embed
        x = x.permute(1, 0, 2) 
        x = self.transformer(x)
        return self.classifier(x.mean(dim=0))