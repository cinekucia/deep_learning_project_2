from abc import ABC, abstractmethod
from typing import Callable
import torch
from torch import Tensor
from transformers import ASTFeatureExtractor

from settings import AUDIO_FILE_METADATA
from .utils import load_ast_config

import librosa
import numpy as np
import torchaudio.transforms as T


class BaseProcessor(ABC):
    @abstractmethod
    def __call__(self, features: Tensor, is_train: bool = False) -> Tensor:
        """
        Convert features to correct format.
        """


class ASTProcessor(BaseProcessor):
    def __init__(self, config_dir: str, **kwargs):
        config = load_ast_config(config_dir, **kwargs)
        self.feature_extractor = ASTFeatureExtractor.from_dict(config.to_dict())

    def __call__(self, features: Tensor, is_train: bool = False) -> Tensor:
        numpy_features = features.numpy()
        sr = AUDIO_FILE_METADATA.get("sample_rate", 16000)

        return self.feature_extractor(
            numpy_features,
            return_tensors="pt",
            sampling_rate=sr
        ).input_values


class ASTAugmenterProcessor(ASTProcessor):
    def __init__(self, config_dir: str, **kwargs):
        self.time_stretch = kwargs.pop("time_stretch", None)
        self.freq_mask = kwargs.pop("freq_mask", None)
        self.time_mask = kwargs.pop("time_mask", None)

        super().__init__(config_dir, **kwargs)

    def __call__(self, features: Tensor, is_train: bool = False) -> Tensor:
        mel_spectogram = super().__call__(features)
        if not is_train:
            return mel_spectogram

        mel_spectogram = mel_spectogram.transpose(1, 2)
        if self.time_stretch is not None:
            stretch = T.TimeStretch(n_freq=128)
            rate = 1 + np.clip(np.random.standard_normal((1,)), -1, 1)[
                       0] * self.time_stretch
            seq_length = mel_spectogram.shape[2]

            mel_spectogram = stretch(
                torch.complex(mel_spectogram, torch.zeros(mel_spectogram.shape)), rate
            ).real

            if mel_spectogram.shape[2] > seq_length:
                start_idx = np.random.randint(
                    0, mel_spectogram.shape[2] - seq_length, mel_spectogram.shape[0]
                )
                mel_spectogram = torch.cat([
                    mel_spectogram[i, :, val:val+seq_length].unsqueeze(0)
                    for i, val in enumerate(start_idx)
                ], dim=0)
            elif mel_spectogram.shape[2] < seq_length:
                mel_spectogram = torch.nn.functional.pad(
                    mel_spectogram, (0, seq_length - mel_spectogram.shape[2])
                )

        if self.freq_mask is not None:
            freq_mask = T.FrequencyMasking(freq_mask_param=self.freq_mask)
            mel_spectogram = freq_mask(mel_spectogram)

        if self.time_mask is not None:
            time_mask = T.TimeMasking(time_mask_param=self.time_mask)
            mel_spectogram = time_mask(mel_spectogram)

        return mel_spectogram.transpose(1, 2)


class ASTNormalizedProcessor(ASTProcessor):
    def __call__(self, features: Tensor, is_train: bool = False) -> Tensor:
        return (super().__call__(features) + 0.4722) / (2 * 0.54427)
