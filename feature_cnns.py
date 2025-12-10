from typing import Optional, Union
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
import torch.nn.functional as F
from audio_features import *

# ----------------------------
# Downsampler
# ----------------------------
class DownSample1D(nn.Module):
    def __init__(self, in_channels, factor):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=factor, stride=factor)

    def forward(self, x):
        # x: (batch, channels, seq)
        return self.conv(x)  # (batch, channels, seq // factor)

# ----------------------------
# Feature CNN Submodules
# ----------------------------

class MelCNN(nn.Module):
    def __init__(self, n_mels=80, model_dim=4096, hidden_dim=512, downsample_factor=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, model_dim, kernel_size=1, stride=1, padding=0)
        )
        self.down = DownSample1D(n_mels, downsample_factor)

    def forward(self, x):
        # x: (n_mels, seq_len)
        x = self.down(x.unsqueeze(0))  # (1, n_mels, seq_len)
        x = self.proj(x)  # (1, model_dim, seq_len)
        x = x.transpose(1,2).squeeze(0)  # (seq_len, model_dim)
        return x


class MFCCCNN(nn.Module):
    def __init__(self, n_mfcc=40, model_dim=4096, hidden_dim=512, downsample_factor=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(n_mfcc, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, model_dim, kernel_size=1, stride=1, padding=0)
        )
        self.down = DownSample1D(n_mfcc, downsample_factor)

    def forward(self, x):
        # x: (n_mfcc, seq_len)
        x = self.down(x.unsqueeze(0))  # (1, n_mfcc, seq_len)
        x = self.proj(x)  # (1, model_dim, seq_len)
        x = x.transpose(1,2).squeeze(0)  # (seq_len, model_dim)
        return x


class ZCRCNN(nn.Module):
    def __init__(self, model_dim=4096, hidden_dim=512, downsample_factor=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, model_dim, kernel_size=1, stride=1, padding=0)
        )
        self.down = DownSample1D(1, downsample_factor)

    def forward(self, x):
        # x: (1, seq_len)
        x = self.down(x.unsqueeze(0))  # (1, 1, seq_len)
        x = self.proj(x)  # (1, model_dim, seq_len)
        x = x.transpose(1,2).squeeze(0)  # (seq_len, model_dim)
        return x


# ----------------------------
# Main Audio Feature CNN
# ----------------------------

class AudioFeatureCNN(nn.Module):
    def __init__(self, model_dim=4096, n_mels=80, n_mfcc=40, downsample_factor=16):
        super().__init__()
        self.mel_proj = MelCNN(n_mels=n_mels, model_dim=model_dim, downsample_factor=downsample_factor)
        self.mfcc_proj = MFCCCNN(n_mfcc=n_mfcc, model_dim=model_dim, downsample_factor=downsample_factor)
        self.zcr_proj = ZCRCNN(model_dim=model_dim, downsample_factor=downsample_factor)
    
    def forward(self, x):
        """
        x: audio waveform (batch, time)
        returns: (batch, combined_seq_len, model_dim)
        """
        device = next(self.parameters()).device  # get device of model weights

        # extract features
        mel = mel_spectrogram(x, n_mels=80, hop_length = 160, win_length = 400).to(torch.bfloat16).to(device)                                    # (batch, n_mels, seq_len_mel)
        mfcc_feat = mfcc(x, n_mfcc=40, n_mels=80, hop_length = 160, win_length = 400).to(torch.bfloat16).to(device)                              # (batch, n_mfcc, seq_len_mfcc)
        zcr_feat = zero_crossing_rate(x, zero_handling="positive", frame_length = 400, hop_length = 160).to(torch.bfloat16).to(device)           # (batch, 1, seq_len_zcr)

        # project each feature
        mel_proj = self.mel_proj(mel)  
        mfcc_proj = self.mfcc_proj(mfcc_feat)
        zcr_proj = self.zcr_proj(zcr_feat)
        
        # concatenate along sequence dimension
        combined = torch.cat([mel_proj, mfcc_proj, zcr_proj], dim=0)
        
        return combined.unsqueeze(0)