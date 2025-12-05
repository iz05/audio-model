from typing import Optional, Union
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
import torch.nn.functional as F
from audio_features import *

# ----------------------------
# Feature Projection Submodules
# ----------------------------

class MelProject(nn.Module):
    def __init__(self, n_mels=80, model_dim=4096, hidden_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_mels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim)
        )
    
    def forward(self, x):
        # x: (batch, n_mels, seq_len)
        x = x.transpose(1,2)  # (batch, seq_len, n_mels)
        return self.proj(x)  # (batch, seq_len, model_dim)

class MFCCProject(nn.Module):
    def __init__(self, n_mfcc=40, model_dim=4096, hidden_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_mfcc, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim)
        )
    
    def forward(self, x):
        x = x.transpose(1,2)  # (batch, seq_len, n_mfcc)
        return self.proj(x)  # (batch, seq_len, model_dim)

class ZCRProject(nn.Module):
    def __init__(self, model_dim=4096, hidden_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim),
        )
    
    def forward(self, x):
        x = x.transpose(1,2)  # (batch, seq_len, 1)
        return self.proj(x)  # (batch, seq_len, model_dim)

# ----------------------------
# Main Audio Feature Projector
# ----------------------------

class AudioFeatureProject(nn.Module):
    def __init__(self, model_dim=4096, n_mels=80, n_mfcc=40):
        super().__init__()
        self.mel_proj = MelProject(n_mels=n_mels, model_dim=model_dim)
        self.mfcc_proj = MFCCProject(n_mfcc=n_mfcc, model_dim=model_dim)
        self.zcr_proj = ZCRProject(model_dim=model_dim)
    
    def forward(self, x):
        """
        x: audio waveform (batch, time)
        returns: (batch, combined_seq_len, model_dim) suitable for LLM
        """
        # extract features
        mel = mel_spectrogram(x, n_mels=80)        # (batch, n_mels, seq_len_mel)
        mfcc_feat = mfcc(x, n_mfcc=40, n_mels=80)  # (batch, n_mfcc, seq_len_mfcc)
        zcr_feat = zero_crossing_rate(x)           # (batch, 1, seq_len_zcr)

        # project each feature
        mel_proj = self.mel_proj(mel)              # (batch, seq_len_mel, model_dim)
        mfcc_proj = self.mfcc_proj(mfcc_feat)
        zcr_proj = self.zcr_proj(zcr_feat)
        
        # concatenate along the sequence dimension
        combined = torch.cat([mel_proj, mfcc_proj, zcr_proj], dim=1)  # (batch, total_seq_len, model_dim)
        
        return combined