from typing import Optional, Union
import numpy as np
import torch
import torchaudio
import librosa

def mel_spectrogram(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 400,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    fmin: int = 0,
    fmax: Optional[int] = None,
    center: bool = True,
) -> torch.Tensor:
    """
    Extract mel spectrogram features from audio.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio waveform.
        sample_rate (int): Sample rate of the input audio. (Default: 16000)
        n_mels (int): Number of Mel bands to generate (for mel_spectrogram). (Default: 128)
        n_fft (int): Size of FFT. (Default: 400)
        hop_length (Optional[int]): Hop length for STFT. If None, it will be set to win_length // 2.
        win_length (Optional[int]): Window length for STFT. If None, it will be set to n_fft.
        fmin (int): Minimum frequency for Mel filter bank. (Default: 0)
        fmax (Optional[int]): Maximum frequency for Mel filter bank. If None, it will be set to sample_rate // 2.
        center (bool): Whether to pad the input on both sides so that the t-th frame is centered at time t * hop_length. (Default: True)

    Returns:
        torch.Tensor: Extracted mel spectrogram features.
    """
    try:
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)

        device = audio.device
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length or (win_length or n_fft) // 2,
            win_length=win_length or n_fft,
            f_min=fmin,
            f_max=fmax or sample_rate // 2,
            center=center,
            power=2.0,
        ).to(device)

        features = mel_spectrogram_transform(audio)
        features = torchaudio.functional.amplitude_to_DB(features, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
        return features
    except Exception as e:
        raise Exception(f"Error occurred during mel spectrogram extraction: {e}")

def mfcc(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 400,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_mels: int = 128,
    fmin: int = 0,
    fmax: Optional[int] = None,
    center: bool = True,
) -> torch.Tensor:
    """
    Extract MFCC features from audio.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio waveform.
        sample_rate (int): Sample rate of the input audio. (Default: 16000)
        n_mfcc (int): Number of MFCCs to return. (Default: 40)
        n_fft (int): Size of FFT. (Default: 400)
        hop_length (Optional[int]): Hop length for STFT. If None, it will be set to win_length // 2.
        win_length (Optional[int]): Window length for STFT. If None, it will be set to n_fft.
        n_mels (int): Number of Mel bands to generate. (Default: 128)
        fmin (int): Minimum frequency for Mel filter bank. (Default: 0)
        fmax (Optional[int]): Maximum frequency for Mel filter bank. If None, it will be set to sample_rate // 2.
        center (bool): Whether to pad the input on both sides so that the t-th frame is centered at time t * hop_length. (Default: True)

    Returns:
        torch.Tensor: Extracted MFCC features.
    """
    try:
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)

        device = audio.device
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length or (win_length or n_fft) // 2,
                'win_length': win_length or n_fft,
                'n_mels': n_mels,
                'f_min': fmin,
                'f_max': fmax or sample_rate // 2,
                'center': center,
            }
        ).to(device)
        features = mfcc_transform(audio)
        return features
    except Exception as e:
        raise Exception(f"Error occurred during MFCC extraction: {e}")

def zero_crossing_rate(
    audio: Union[np.ndarray, torch.Tensor],
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Extract zero crossing rate features (ZCR) from audio.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio waveform.
        frame_length (int): Length of the analysis frame.
        hop_length (int): Hop length for analysis.
        center (bool): Whether to pad the input on both sides so that the t-th frame is centered at time t * hop_length. (Default: True)
        threshold (float): Threshold for considering a zero crossing. (Default: 0.0)

    Returns:
        torch.Tensor: Extracted zero crossing rate features.
        Note: the output has shape (..., 1, T) where T is the number of frames.

    WARNING: This function uses librosa and may not be optimized for GPU usage.
    WARNING: Librosa clips all values within threshold to 0, but treats 0 as positive.
             This may lead to behavior unoptimized for audio, particularly human speech.
    """
    try:
        device = audio.device if isinstance(audio, torch.Tensor) else torch.device('cpu')
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        features = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
            center=center,
            threshold=threshold,
        )
        return torch.Tensor(features).to(device)
    except Exception as e:
        raise Exception(f"Error occurred during zero crossing rate extraction: {e}")

# TODO: Add other feature extraction methods like RMSE if needed