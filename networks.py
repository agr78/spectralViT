import torch
import torch.nn as nn

class SpectralAttention(nn.Module):
    def __init__(self, n_components, embed_dim=16, n_heads=4):
        super().__init__()
        self.n_components = n_components
        self.input_projection = nn.Linear(1, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, n_heads)
        self.output_projection = nn.Linear(embed_dim, 1)
        
        # Fixed variance-based weighting (The Hierarchy Prior)
        indices = torch.arange(n_components).float()
        # Decay factor: PCA 1 stays strong, PCA 128 is suppressed
        self.register_buffer('hierarchy_weight', torch.exp(-indices / 32).view(1, -1, 1))

    def forward(self, x):
        pca_raw = x[:, :self.n_components].unsqueeze(-1) 
        clin_feats = x[:, self.n_components:]
        
        # 1. Project with hierarchy weighting
        h = self.input_projection(pca_raw * self.hierarchy_weight) 
        
        # 2. Attention mechanism
        h = h.transpose(0, 1) 
        attn_out, _ = self.mha(h, h, h)
        h_out = attn_out.transpose(0, 1)
        
        # 3. Residual spectral context (Nonlinear refinement)
        spectral_context = self.output_projection(h_out)
        # We only add 10% of the attention's "opinion" to keep the VAE stable
        pca_out = (pca_raw + 0.1 * spectral_context).squeeze(-1) 
        
        return torch.cat([pca_out, clin_feats], dim=1)

class SpatialViT(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_channels=1, num_classes=1, 
                 dim=128, depth=4, heads=8, mlp_dim=256, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by patch size."
        num_patches = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        # REMOVED batch_first=True for backward compatibility
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, 
            dropout=dropout, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Sigmoid()
        )
    def forward(self, x, mode='fine_tune'):
        p = self.to_patch_embedding(x).transpose(1, 2) # [Batch, Seq, Dim]
        b, n, _ = p.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, p), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        # MANUALLY TRANSPOSE: Transformer expects [Seq, Batch, Dim]
        x = x.transpose(0, 1) 
        x = self.transformer(x)
        x = x.transpose(0, 1) # Transpose back to [Batch, Seq, Dim]
        cls_out = x[:, 0]
        if mode == 'pretrain':
            return None, cls_out # Decoder is handled externally in your script
        return self.mlp_head(cls_out).squeeze(-1)
    
class FullSpectralAttention(nn.Module):
    def __init__(self, n_components, embed_dim=16, n_heads=4):
        super().__init__()
        self.n_components = n_components
        self.input_projection = nn.Linear(1, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, 1)

    def forward(self, x):
        pca_feats = x[:, :self.n_components].unsqueeze(-1) 
        clin_feats = x[:, self.n_components:]
        h = self.input_projection(pca_feats) 
        h = h.transpose(0, 1) 
        attn_out, _ = self.mha(h, h, h)
        h = self.norm(h + attn_out)
        h = h.transpose(0, 1) 
        pca_out = self.output_projection(h).squeeze(-1) 
        return torch.cat([pca_out, clin_feats], dim=1)

class HybridAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64))
        self.decoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim))
        self.classifier_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x, mode='pretrain'):
        latent = self.encoder(x)
        return self.decoder(latent) if mode == 'pretrain' else self.classifier_head(latent).squeeze()

class SupervisedManifoldAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, input_dim))
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x, mode='train'):
        latent = self.encoder(x)
        return self.decoder(latent) if mode == 'pretrain' else self.classifier(latent).squeeze()

class SupervisedManifoldVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, input_dim))
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mode='train'):
        h = self.encoder_base(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        if mode == 'pretrain':
            return self.decoder(z), mu, logvar
        else:
            return self.classifier(mu).squeeze() # Use mu for stable inference

class QSMDecoder(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        # Input is (B, 512, 1, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, 256, 4, 1, 0), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.decoder(x)