from typing import Optional, Union
import numpy as np
import torch
import torchaudio

def mel_spectrogram(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    fmin: int = 0,
    fmax: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract mel spectrogram features from audio.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio waveform.
        feature_type (str): Type of feature to extract. Options are "mel_spectrogram", "mfcc", "spectrogram".
        sample_rate (int): Sample rate of the input audio.
        n_mels (int): Number of Mel bands to generate (for mel_spectrogram).
        n_fft (int): Size of FFT.
        hop_length (int): Hop length for STFT.
        win_length (int): Window length for STFT.
        fmin (int): Minimum frequency for Mel filter bank.
        fmax (Optional[int]): Maximum frequency for Mel filter bank. If None, it will be set to sample_rate // 2.

    Returns:
        torch.Tensor: Extracted audio features.
    """
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio, dtype=torch.float32)

    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        f_min=fmin,
        f_max=fmax or sample_rate // 2,
    )
    features = mel_spectrogram_transform(audio)
    features = torchaudio.functional.amplitude_to_DB(features, multiplier=10.0, amin=1e-10, db_multiplier=0.0)

# TODO: Add other feature extraction methods like MFCC if needed