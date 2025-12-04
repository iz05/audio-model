from typing import Optional, Union, Any
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

def forward_fill(
    input: torch.Tensor,
    to_fill: Any = 0,
) -> torch.Tensor:
    """
    Forward fill values in a tensor.

    Args:
        input (torch.Tensor): Input tensor with potential NaN values.
        to_fill (Any): Value to consider as missing (to be filled). (Default: 0)

    Returns:
        torch.Tensor: Tensor with to_fill values forward filled.
    """
    
    l = input.size(-1)
    stack = torch.stack([input.clone() for _ in range(l)], dim=-2)
    tril_mask = torch.ones_like(stack, dtype=bool).tril()
    stack = stack.where(tril_mask, to_fill)

    if isinstance(to_fill, float) and np.isnan(to_fill):
        mask = (torch.isnan(stack)).logical_not()
    else:
        mask = (stack == to_fill).logical_not()

    indices = l - 1 - mask.flip(-1).int().argmax(dim=-1, keepdim=True)
    return stack.gather(dim=-1, index = indices).squeeze(-1)

def backward_fill(
    input: torch.Tensor,
    to_fill: Any = 0,
) -> torch.Tensor:
    """
    Backward fill values in a tensor.

    Args:
        input (torch.Tensor): Input tensor with potential NaN values.
        to_fill (Any): Value to consider as missing (to be filled). (Default: 0)

    Returns:
        torch.Tensor: Tensor with to_fill values backward filled.
    """
    
    l = input.size(-1)
    stack = torch.stack([input.clone() for _ in range(l)], dim=-2)
    triu_mask = torch.ones_like(stack, dtype=bool).triu()
    stack = stack.where(triu_mask, to_fill)

    if isinstance(to_fill, float) and np.isnan(to_fill):
        mask = (torch.isnan(stack)).logical_not()
    else:
        mask = (stack == to_fill).logical_not()

    indices = mask.int().argmax(dim=-1, keepdim=True)
    return stack.gather(dim=-1, index = indices).squeeze(-1)

def zero_crossing_rate(
    audio: Union[np.ndarray, torch.Tensor],
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    threshold: float = 0.0,
    zero_handling: str = 'forward_fill',
) -> torch.Tensor:
    """
    Extract zero crossing rate features (ZCR) from audio.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): Input audio waveform.
        frame_length (int): Length of the analysis frame. (Default: 2048)
        hop_length (int): Hop length for analysis. (Default: 512)
        center (bool): Whether to pad the input on both sides so that the t-th frame is centered at time t * hop_length. (Default: True)
        threshold (float): Threshold for considering a zero crossing. (Default: 0.0)
        zero_handling (str): Method to handle silent frames. Options are listed below. (Default: 'forward_fill')
            - 'forward_fill': Fill silent frames with the last valid value.
            - 'backward_fill': Fill silent frames with the next valid value.
            - 'positive': Treat silent frames as positive.
            - 'negative': Treat silent frames as negative.
            - 'zero': Treat silent frames as zero, their own sign (Both + -> 0 and - -> 0 count as sign crossings.)

    Returns:
        torch.Tensor: Extracted zero crossing rate features.
        Note: the output has shape (..., 1, T) where T is the number of frames.

    WARNING: This function uses librosa and may not be optimized for GPU usage.
    """
    try:
        if zero_handling not in ['forward_fill', 'backward_fill', 'positive', 'negative', 'zero']:
            raise ValueError(f"Invalid zero_handling method: {zero_handling}")
        
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        clipped_audio = audio.where(audio.abs() >= threshold, 0.0)
        if zero_handling == 'forward_fill':
            clipped_audio = forward_fill(clipped_audio, to_fill=0.0)
        elif zero_handling == 'backward_fill':
            clipped_audio = backward_fill(clipped_audio, to_fill=0.0)
        elif zero_handling == 'positive':
            clipped_audio = clipped_audio.where(clipped_audio != 0.0, threshold + 1)
        elif zero_handling == 'negative':
            clipped_audio = clipped_audio.where(clipped_audio != 0.0, -threshold - 1)

        device = clipped_audio.device
        clipped_audio = clipped_audio.cpu().numpy()

        features = librosa.feature.zero_crossing_rate(
            y=clipped_audio,
            frame_length=frame_length,
            hop_length=hop_length,
            center=center,
            threshold=threshold,
            zero_pos=False,
        )
        return torch.Tensor(features).to(device)
    except Exception as e:
        raise Exception(f"Error occurred during zero crossing rate extraction: {e}")

# TODO: Add other feature extraction methods like RMSE if needed