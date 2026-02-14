import torch
import torch.nn as nn

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

class ResNetWrapper(nn.Module):
    def __init__(self, model, clinical_dim):
        super().__init__()
        self.base_model = model
        self.feat_dim = model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_dim + clinical_dim, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, 2)
        )
    def forward(self, x, clinical_vec):
        feats = self.base_model(x)
        logits = self.fusion(torch.cat([feats, clinical_vec], dim=1))
        return logits, feats.view(feats.size(0), self.feat_dim, 1, 1)