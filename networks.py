import torch
import torch.nn as nn
import torch
import torch.nn as nn

class ClinicalTransformer(nn.Module):
    def __init__(self, n_inputs, embed_dim=32, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(embed_dim * n_inputs, 1)

    def forward(self, x, return_logit=False, **kwargs):
        if x.ndim == 1: x = x.unsqueeze(0)
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2).flatten(1)
        logit = self.fc(x).squeeze()
        if logit.ndim == 0: logit = logit.unsqueeze(0) # handle single-sample batch
        return logit if return_logit else torch.sigmoid(logit)

class SequentialSpectralAttention(nn.Module):
    def __init__(self, n_components, n_clinical, embed_dim=32, n_heads=4, tau_0=16, alpha_0=0.1):
        super().__init__()
        self.n_components = n_components
        self.n_clinical = n_clinical
        self.tau_0 = tau_0
        self.alpha_0 = alpha_0
        
        # Projections
        self.input_projection = nn.Linear(1, embed_dim)
        self.clin_embedders = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_clinical)])
        # Step 1: Spectral Self-Attention (Modeling inter-component dependencies)
        self.self_attn_spec = nn.MultiheadAttention(embed_dim, n_heads)
        # Step 2: Cross-Attention (Clinical queries the refined Spectral library)
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads)
        # Projects attended context back to match clinical input scalar
        self.output_projection = nn.Linear(embed_dim, 1)
        # Learnable hierarchy weight for PCA components
        self.hierarchy_weight = nn.Parameter(
            torch.exp(-torch.arange(n_components).float() / self.tau_0).view(1, -1, 1)
        )
        # Learnable gating parameter for the clinical update
        self.gate = nn.Parameter(torch.tensor([self.alpha_0]))

    def forward(self, x, skip_clin=False):
        B = x.shape[0]
        # --- PHASE 1: Spectral Manifold Encoding ---
        # Prepare Spectral Library
        pca_raw = x[:, :self.n_components].unsqueeze(-1) # [B, N_comp, 1]
        h_spec = self.input_projection(pca_raw * self.hierarchy_weight) # [B, N_comp, Dim]
        # Spectral Self-Attention (B, N, E) -> (N, B, E)
        h_spec_t = h_spec.transpose(0, 1)
        # Components "talk" to each other to capture non-linear anatomical context
        h_spec_refined_t, _ = self.self_attn_spec(h_spec_t, h_spec_t, h_spec_t)
        # If no clinical data, return the spectral path only (or as configured)
        if skip_clin or self.n_clinical == 0:
            # Note: You may want to project h_spec_refined back to 1D if skipping
            return x
        # --- PHASE 2: Clinical-Gated Extraction ---
        # Prepare Clinical Path (Queries)
        clin_raw = x[:, self.n_components:] # [B, N_clin]
        h_clin = torch.stack([
            self.clin_embedders[i](clin_raw[:, i:i+1]) for i in range(self.n_clinical)
        ], dim=1) # [B, N_clin, Dim]
        # Cross-Attention: Clinical (Q) attends to Refined Spectral (K, V)
        q = h_clin.transpose(0, 1)           # [N_clin, B, Dim]
        k = v = h_spec_refined_t             # [N_comp, B, Dim]
        # Resulting attn_out_t has shape [N_clin, B, Dim]
        attn_out_t, _ = self.cross_attn(query=q, key=k, value=v)
        h_out = attn_out_t.transpose(0, 1)    # [B, N_clin, Dim]
        # --- PHASE 3: Gated Residual Update ---
        # Project attended context back to scalar dimension for residual addition
        clinical_context = self.output_projection(h_out).squeeze(-1) # [B, N_clin]
        # Update only the clinical portion of the feature vector
        clin_out = clin_raw + self.gate * clinical_context
        # Final output: [Original PCA components, Updated Clinical variables]
        return torch.cat([x[:, :self.n_components], clin_out], dim=1)

class GatedCrossSpectralAttention(nn.Module):
    def __init__(self, n_components, n_clinical, embed_dim=32, n_heads=4, tau_0=16, alpha_0=0.1):
        super().__init__()
        self.n_components = n_components
        self.n_clinical = n_clinical
        self.tau_0 = tau_0
        self.alpha_0 = alpha_0
        
        # Projections
        self.input_projection = nn.Linear(1, embed_dim)
        self.clin_embedders = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_clinical)])
        
        # Cross-Attention: Clinical queries the Spectral library
        self.mha = nn.MultiheadAttention(embed_dim, n_heads)
        
        # Projects attended context back to match clinical input (1 feature per token)
        self.output_projection = nn.Linear(embed_dim, 1)
        
        # Learnable hierarchy weight for PCA components
        self.hierarchy_weight = nn.Parameter(
            torch.exp(-torch.arange(n_components).float() / self.tau_0).view(1, -1, 1)
        )
        
        # Learnable gating parameter
        self.gate = nn.Parameter(torch.tensor([self.alpha_0]))

    def forward(self, x, skip_clin=False):
        B = x.shape[0]
        
        # 1. Prepare Spectral Library (Keys & Values)
        pca_raw = x[:, :self.n_components].unsqueeze(-1) # [B, N_comp, 1]
        h_spec = self.input_projection(pca_raw * self.hierarchy_weight) # [B, N_comp, Dim]
        
        # If no clinical data, cross-attention cannot occur; return original x
        if skip_clin or self.n_clinical == 0:
            return x

        # 2. Prepare Clinical Path (Queries)
        clin_raw = x[:, self.n_components:] # [B, N_clin]
        h_clin = torch.stack([
            self.clin_embedders[i](clin_raw[:, i:i+1]) for i in range(self.n_clinical)
        ], dim=1) # [B, N_clin, Dim]
        
        # 3. Cross-Attention: Clinical (Q) attends to Spectral (K, V)
        # MultiheadAttention expects (S, B, E)
        q = h_clin.transpose(0, 1)      # [N_clin, B, Dim]
        k = v = h_spec.transpose(0, 1)  # [N_comp, B, Dim]
        
        # Resulting attn_out_t has shape [N_clin, B, Dim]
        attn_out_t, _ = self.mha(query=q, key=k, value=v)
        h_out = attn_out_t.transpose(0, 1) # [B, N_clin, Dim]
        
        # 4. Gated Residual Update on Clinical Path
        # The context is now a clinical-aligned representation of spectral data
        clinical_context = self.output_projection(h_out).squeeze(-1) # [B, N_clin]
        
        # Update the clinical portion of the vector
        clin_out = clin_raw + self.gate * clinical_context
        
        # Return PCA components (unchanged) concatenated with updated clinical data
        return torch.cat([x[:, :self.n_components], clin_out], dim=1)

class GatedSpectralAttention(nn.Module):
    def __init__(self, n_components, n_clinical, embed_dim=32, n_heads=4, tau_0=16, alpha_0=0.1):
        super().__init__()
        self.n_components = n_components
        self.n_clinical = n_clinical
        self.tau_0 = tau_0
        self.alpha_0 = alpha_0
        self.input_projection = nn.Linear(1, embed_dim)
        self.clin_embedders = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_clinical)])
        
        self.mha = nn.MultiheadAttention(embed_dim, n_heads)
        self.output_projection = nn.Linear(embed_dim, 1)
        
        # Learnable hierarchy weight
        self.hierarchy_weight = nn.Parameter(torch.exp(-torch.arange(n_components).float() / self.tau_0).view(1, -1, 1))
        
        # Learnable gating parameter initialized at 0.1
        self.gate = nn.Parameter(torch.tensor([self.alpha_0]))

    def forward(self, x, skip_clin=False):
        B = x.shape[0]
        pca_raw = x[:, :self.n_components].unsqueeze(-1) # [B, N, 1]
        h_spec = self.input_projection(pca_raw * self.hierarchy_weight) # [B, N, Dim]
        
        if not skip_clin:
            clin_raw = x[:, self.n_components:] 
            h_clin = torch.stack([self.clin_embedders[i](clin_raw[:, i:i+1]) for i in range(self.n_clinical)], dim=1)
            tokens = torch.cat([h_spec, h_clin], dim=1)
        else:
            tokens = h_spec
            
        # MHA Manual Transpose (B, S, E) -> (S, B, E)
        tokens_t = tokens.transpose(0, 1)
        attn_out_t, _ = self.mha(tokens_t, tokens_t, tokens_t)
        h_out = attn_out_t.transpose(0, 1)
        
        # Extract PCA context
        spectral_context = self.output_projection(h_out[:, :self.n_components, :])
        
        # Use learnable gate instead of fixed 0.1
        pca_out = (pca_raw + self.gate * spectral_context).squeeze(-1)
        
        if skip_clin: return pca_out
        return torch.cat([pca_out, x[:, self.n_components:]], dim=1)

class EnhancedSpatialAttention(nn.Module):
    def __init__(self, img_dim=64, patch_size=16, n_clinical=6, embed_dim=16, n_heads=4, target_dim=128):
        super().__init__()
        self.img_dim = img_dim
        self.patch_size = patch_size
        self.n_patches = (img_dim // patch_size) ** 2 
        self.n_clinical = n_clinical
        
        self.patch_projection = nn.Linear(patch_size**2, embed_dim)
        self.clin_embedders = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_clinical)])
        
        # Positional embeddings for patches + clinical tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + n_clinical, embed_dim))
        
        # Standard MHA (No batch_first for compatibility)
        self.mha = nn.MultiheadAttention(embed_dim, n_heads)
        self.bottleneck = nn.Linear(self.n_patches * embed_dim, target_dim)

    def forward(self, x, skip_clin=False):
        B = x.shape[0]
        
        # 1. Image Tokenization
        x_img_raw = x[:, :self.img_dim**2]
        x_img = x_img_raw.view(B, 1, self.img_dim, self.img_dim)
        patches = x_img.unfold(2, self.patch_size, self.patch_size)\
                       .unfold(3, self.patch_size, self.patch_size)\
                       .contiguous().view(B, self.n_patches, -1)
        h_patches = self.patch_projection(patches) # (B, n_patches, embed_dim)
        
        if not skip_clin:
            # 2. Atomic Clinical Tokenization
            x_clin = x[:, self.img_dim**2:]
            h_clin = torch.stack([self.clin_embedders[i](x_clin[:, i:i+1]) for i in range(self.n_clinical)], dim=1)
            tokens = torch.cat([h_clin, h_patches], dim=1)
            tokens = tokens + self.pos_embedding
        else:
            tokens = h_patches + self.pos_embedding[:, self.n_clinical:, :]
        
        # 3. MHA with Manual Transpose (B, S, E) -> (S, B, E)
        tokens_t = tokens.transpose(0, 1)
        attn_out_t, _ = self.mha(tokens_t, tokens_t, tokens_t)
        h_out = attn_out_t.transpose(0, 1) # Back to (B, S, E)
        
        # 4. Extract Patches
        start_idx = self.n_clinical if not skip_clin else 0
        spatial_refined = h_out[:, start_idx:, :].reshape(B, -1)
        spatial_compressed = self.bottleneck(spatial_refined)
        
        if skip_clin: return spatial_compressed
        return torch.cat([spatial_compressed, x_clin], dim=1)

class EnhancedSpectralAttention(nn.Module):
    def __init__(self, n_components, n_clinical, embed_dim=32, n_heads=4):
        super().__init__()
        self.n_components = n_components
        self.n_clinical = n_clinical
        
        self.input_projection = nn.Linear(1, embed_dim)
        self.clin_embedders = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_clinical)])
        
        self.mha = nn.MultiheadAttention(embed_dim, n_heads)
        self.output_projection = nn.Linear(embed_dim, 1)
        
        # Learnable hierarchy weight
        self.hierarchy_weight = nn.Parameter(torch.exp(-torch.arange(n_components).float() / 16).view(1, -1, 1))

    def forward(self, x, skip_clin=False):
        B = x.shape[0]
        pca_raw = x[:, :self.n_components].unsqueeze(-1) # [B, N, 1]
        h_spec = self.input_projection(pca_raw * self.hierarchy_weight) # [B, N, Dim]
        
        if not skip_clin:
            clin_raw = x[:, self.n_components:] 
            h_clin = torch.stack([self.clin_embedders[i](clin_raw[:, i:i+1]) for i in range(self.n_clinical)], dim=1)
            tokens = torch.cat([h_spec, h_clin], dim=1)
        else:
            tokens = h_spec
            
        # MHA Manual Transpose (B, S, E) -> (S, B, E)
        tokens_t = tokens.transpose(0, 1)
        attn_out_t, _ = self.mha(tokens_t, tokens_t, tokens_t)
        h_out = attn_out_t.transpose(0, 1)
        
        # Extract PCA context
        spectral_context = self.output_projection(h_out[:, :self.n_components, :])
        pca_out = (pca_raw + 0.1 * spectral_context).squeeze(-1)
        
        if skip_clin: return pca_out
        return torch.cat([pca_out, x[:, self.n_components:]], dim=1)

class MinimalSpectralAttention(nn.Module):
    def __init__(self, n_components, n_clinical, embed_dim=16, n_heads=4):
        super().__init__()
        self.n_components = n_components
        
        # Projections for both types of tokens
        self.input_projection = nn.Linear(1, embed_dim)
        self.clin_projection = nn.Linear(n_clinical, embed_dim) # New: Clinical Projector
        
        self.mha = nn.MultiheadAttention(embed_dim, n_heads)
        self.output_projection = nn.Linear(embed_dim, 1)
        
        indices = torch.arange(n_components).float()
        self.register_buffer('hierarchy_weight', torch.exp(-indices / 32).view(1, -1, 1))

    def forward(self, x, skip_clin=False):
        if skip_clin:
            # FIX: Only take the first 128 dimensions (PCA) for hierarchy weighting
            pca_raw = x[:, :self.n_components].unsqueeze(-1) 
            h_spec = self.input_projection(pca_raw * self.hierarchy_weight)
            h = h_spec.transpose(0, 1)
            attn_out, _ = self.mha(h, h, h)
            # Return only the PCA part (128)
            return (pca_raw + 0.1 * self.output_projection(attn_out.transpose(0, 1))).squeeze(-1)
        # 1. Split data
        pca_raw = x[:, :self.n_components].unsqueeze(-1) 
        clin_raw = x[:, self.n_components:]
        
        # 2. Project Spectral Tokens
        h_spec = self.input_projection(pca_raw * self.hierarchy_weight) # (B, N_PCA, Dim)
        
        # 3. Project Clinical Data into a single "Global Context Token"
        h_clin = self.clin_projection(clin_raw).unsqueeze(1) # (B, 1, Dim)
        
        # 4. Concatenate: Now the sequence is [Spectral_1, ..., Spectral_N, Clinical_Token]
        combined_tokens = torch.cat([h_spec, h_clin], dim=1)
        
        # 5. Multi-Head Attention (Self-Attention across all tokens)
        h = combined_tokens.transpose(0, 1) 
        attn_out, _ = self.mha(h, h, h)
        h_out = attn_out.transpose(0, 1)
        
        # 6. Extract the Spectral part back out and project back to raw PCA space
        spectral_context = self.output_projection(h_out[:, :self.n_components, :])
        pca_out = (pca_raw + 0.1 * spectral_context).squeeze(-1) 
        
        # We return the original clinical features + the refined PCA
        return torch.cat([pca_out, clin_raw], dim=1)
    
class MiniSpatialAttention(nn.Module):
    def __init__(self, img_dim=64, patch_size=16, n_clinical=5, embed_dim=16, n_heads=4, target_dim=128):
        super().__init__()
        self.img_dim = img_dim
        self.patch_size = patch_size
        self.n_patches = (img_dim // patch_size) ** 2 
        
        # Projections
        self.patch_projection = nn.Linear(patch_size**2, embed_dim)
        self.clin_projection = nn.Linear(n_clinical, embed_dim) # Clinical -> Embedding
        
        # Spatial Awareness
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim)) # +1 for clinical token
        self.mha = nn.MultiheadAttention(embed_dim, n_heads)
        
        # Bottleneck to keep VAE input consistent (128 + n_clinical)
        self.bottleneck = nn.Linear(self.n_patches * embed_dim, target_dim)

    def forward(self, x, skip_clin=False):
        B = x.shape[0]
        if skip_clin:
            # Pretraining Mode: x is [B, Pixels]
            x_img_raw = x[:, :self.img_dim**2] 
            x_img = x_img_raw.view(B, 1, self.img_dim, self.img_dim)
            patches = x_img.unfold(2, self.patch_size, self.patch_size)\
                           .unfold(3, self.patch_size, self.patch_size)\
                           .contiguous().view(B, self.n_patches, -1)
            h_patches = self.patch_projection(patches)
            # Use only patch pos embeddings (indices 1 to N)
            combined = h_patches + self.pos_embedding[:, 1:, :]
            h = combined.transpose(0, 1)
            attn_out, _ = self.mha(h, h, h)
            spatial_refined = attn_out.transpose(0, 1).reshape(B, -1)
            return self.bottleneck(spatial_refined)
        # 1. Unpack Raw Pixels and Clinical Data
        x_img_raw = x[:, :self.img_dim**2]
        x_clin = x[:, self.img_dim**2:]
        B = x_img_raw.shape[0]
        
        # 2. Tokenize Image Patches
        x_img = x_img_raw.view(B, 1, self.img_dim, self.img_dim)
        patches = x_img.unfold(2, self.patch_size, self.patch_size)\
                       .unfold(3, self.patch_size, self.patch_size)\
                       .contiguous().view(B, self.n_patches, -1)
        h_patches = self.patch_projection(patches) # (B, n_patches, embed_dim)
        
        # 3. Project Clinical Context Token
        h_clin = self.clin_projection(x_clin).unsqueeze(1) # (B, 1, embed_dim)
        
        # 4. Concatenate: [Clinical Token, Patch 1, ..., Patch N]
        # Adding position embeddings to everything
        combined = torch.cat([h_clin, h_patches], dim=1) + self.pos_embedding
        
        # 5. Multi-Head Attention
        # Now every patch "sees" the clinical data during the attention sweep
        h = combined.transpose(0, 1)
        attn_out, _ = self.mha(h, h, h)
        h_out = attn_out.transpose(0, 1)
        
        # 6. Separate and Refine
        # We extract the patches back out to go through the bottleneck
        spatial_refined = h_out[:, 1:, :].reshape(B, -1)
        spatial_compressed = self.bottleneck(spatial_refined)
        
        # Return compressed spatial features + original clinical features for the VAE
        return torch.cat([spatial_compressed, x_clin], dim=1)

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