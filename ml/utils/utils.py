from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
import librosa
import warnings
import torch.nn.functional as F
import torch

import torchaudio.transforms as T


warnings.filterwarnings('ignore')


def preprocess_audio(file_path: Union[str, bytes, Path]) -> torch.Tensor:
    """
    Returns tensor for SciKitCNN model inference:
    shape = (n_segments, 1, image_size, image_size)
    """
    target_sr = 22050
    segment_length = 3
    overlap = 0
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    image_size = 150

    if isinstance(file_path, (str, Path)):
        audio_file = str(file_path)
    elif isinstance(file_path, bytes):
        audio_file = BytesIO(file_path)
    else:
        raise TypeError(
            f"Path or bytes are only valid options, "
            f"input type: {type(file_path)}"
        )

    audio, file_sr = librosa.load(audio_file, sr=target_sr, mono=True)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if file_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=target_sr)

    segment_samples = segment_length * target_sr
    step = (segment_length - overlap) * target_sr

    segments = []

    if len(audio) < segment_samples:
        assert Exception("Too shirt audio length")

    for start in range(0, len(audio) - segment_samples + 1, step):
        end = start + segment_samples
        segments.append(audio[start:end])

    mel_specs = []
    for segment in segments:
        mel_spec = librosa.feature.melspectrogram(
            y=segment,
            sr=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Transform into a tensor for interpolation: [Batch=1, Channel=1, H, W]
        mel_tensor = torch.tensor(
            mel_spec_db,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        # Resize to (150, 150)
        mel_resized = F.interpolate(mel_tensor, size=(image_size, image_size),
                                    mode='bilinear', align_corners=False)

        # Make shape [1, 150, 150]
        mel_specs.append(mel_resized.squeeze(0))

    output_tensor = torch.stack(mel_specs)

    return output_tensor


def preprocess_audio_torch(file_path: Union[str, bytes, Path]) -> torch.Tensor:
    target_sr = 22050
    segment_length = 4
    overlap = 2
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    image_size = 150

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(file_path, (str, Path)):
        audio_file = str(file_path)
    elif isinstance(file_path, bytes):
        audio_file = BytesIO(file_path)
    else:
        raise TypeError(
            f"Path or bytes are only valid options, "
            f"input type: {type(file_path)}"
        )

    audio, file_sr = librosa.load(audio_file, sr=target_sr, mono=True)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if file_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=target_sr)

    segment_samples = segment_length * target_sr
    step = (segment_length - overlap) * target_sr

    segments = []

    if len(audio) < segment_samples:
        raise ValueError("Too short audio length")

    for start in range(0, len(audio) - segment_samples + 1, step):
        end = start + segment_samples
        segments.append(audio[start:end])

    mel_transform = T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
        normalized=True
    ).to(device)

    amplitude_to_db = T.AmplitudeToDB(stype='power').to(device)

    mel_specs = []

    for segment in segments:
        waveform = torch.tensor(segment, dtype=torch.float32, device=device)
        mel_spec = mel_transform(waveform)
        mel_spec_db = amplitude_to_db(mel_spec)

        # (n_mels, time) → (1, 1, n_mels, time)
        mel_tensor = mel_spec_db.unsqueeze(0).unsqueeze(0)

        mel_resized = F.interpolate(
            mel_tensor,
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False
        )

        mel_specs.append(mel_resized.squeeze(0).cpu())

    # [n_segments, 1, 150, 150]
    output_tensor = torch.stack(mel_specs)

    return output_tensor
