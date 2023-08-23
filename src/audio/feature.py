import platform

import numpy as np
import torch
from torch import FloatTensor, Tensor


class Spectrogram(object):
    """
    Create a spectrogram from a audio signal
    STFT(Short Time Fourier Transform)
        - 음성을 잘게 (0.001초 정도) 잘라서 각 작은 조각에 푸리에 변환을 적용하는 신호처리 방법
        - STFT 결과에 L2 Norm 적용시키면 Spectrogram을 구할 수 있음

    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        sr: int = 16000,
        frame_length: int = 20,
        frame_shift: int = 10,
        feature_extract_by: str = "torch",  # [torch, kaldi]
    ) -> None:
        self.sr = sr
        self.feature_extract_by = feature_extract_by.lower()

        if self.feature_extract_by == "kaldi":
            # torch audio에서 kaldi의 기능을 쓸 수 있음
            # torchaudio only supported on Linux(Ubuntu, Mac)
            assert (
                platform.system().lower() == "linux"
                or platform.system().lower() == "darwin"
            )
            try:
                import torchaudio
            except ImportError:
                raise ImportError("Please install torchaudio: `pip install torchaudio`")

            self.transform = torchaudio.compliance.kaldi.spectrogram
            self.frame_length = frame_length
            self.frame_shift = frame_shift

        else:
            self.n_fft = int(round(sr * 0.001 * frame_length))
            self.hop_length = int(round(sr * 0.001 * frame_shift))

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if self.feature_extract_by == "kaldi":
            spectrogram = self.transform(
                Tensor(signal).unsqueeze(1),
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                sample_frequency=self.sr,
            ).transpose(0, 1)

        elif self.feature_extract_by == "torch":  # torch
            spectrogram = torch.stft(
                Tensor(signal), self.n_fft, hop_length=self.hop_length,
                win_length=self.n_fft, window=torch.hamming_window(self.n_fft),
                center=False, normalized=False, onesided=True, return_complex=False
            )
            spectrogram = (spectrogram[:, :, 0].pow(2) + spectrogram[:, :, 1].pow(2)).pow(0.5)
            spectrogram = np.log1p(spectrogram.numpy())
        
        else:
            raise ValueError(
                "Unsupported library : {0}".format(self.feature_extract_by)
            )

        return spectrogram


class MelSpectrogram(object):
    """
    Create MelSpectrogram for a raw audio signal. This is a composition of Spectrogram and MelScale.

    Args:
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mels (int):  Number of mfc coefficients to retain. (Default: 80)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
        feature_extract_by (str): which library to use for feature extraction (default: librosa)
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 80,
        frame_length: int = 20,
        frame_shift: int = 10,
        feature_extract_by: str = "librosa",  # [torch, librosa]
    ) -> None:
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = int(round(sr * 0.001 * frame_length))
        self.hop_length = int(round(sr * 0.001 * frame_shift))
        self.feature_extract_by = feature_extract_by.lower()

        if self.feature_extract_by == "torchaudio":
            # torch audio에서 kaldi의 기능을 쓸 수 있음
            # torchaudio only supported on Linux(Ubuntu, Mac)
            assert (
                platform.system().lower() == "linux"
                or platform.system().lower() == "darwin"
            )
            import torchaudio

            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                win_length=frame_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
            )

        else:
            import librosa

            self.transform = librosa.feature.melspectrogram
            self.power_to_db = librosa.power_to_db

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if self.feature_extract_by == "torchaudio":
            import torchaudio
            mel_spectrogram = self.transform(Tensor(signal))
            mel_spectrogram = torchaudio.functional.amplitude_to_DB(
                mel_spectrogram, multiplier=10, amin=1e-10,
                db_multiplier=torch.log10(
                    torch.max(Tensor([torch.max(mel_spectrogram), 1e-10]))
                    ),
                top_db=80)
            mel_spectrogram = mel_spectrogram.numpy()

        elif self.feature_extract_by == "librosa":
            import librosa
            mel_spectrogram = librosa.feature.melspectrogram(
                y=signal,
                sr=self.sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            mel_spectrogram = self.power_to_db(mel_spectrogram, ref=np.max)

        else:
            raise ValueError(
                "Unsupported library : {0}".format(self.feature_extract_by)
            )
        return mel_spectrogram


class MFCC(object):
    """
    Create the Mel-frequency cepstrum coefficients (MFCCs) from an audio signal.

    Args:
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mfcc (int):  Number of mfc coefficients to retain. (Default: 40)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
        feature_extract_by (str): which library to use for feature extraction(default: librosa)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        frame_length: int = 20,
        frame_shift: int = 10,
        feature_extract_by: str = "librosa",
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = int(round(sample_rate * 0.001 * frame_length))
        self.hop_length = int(round(sample_rate * 0.001 * frame_shift))
        self.feature_extract_by = feature_extract_by.lower()

        if self.feature_extract_by == "torchaudio":
            # torchaudio is only supported on Linux (Linux, Mac)
            assert (
                platform.system().lower() == "linux"
                or platform.system().lower() == "darwin"
            )
            import torchaudio

            self.transforms = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                log_mels=True,
                win_length=frame_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
            )
        else:
            import librosa

            self.transforms = librosa.feature.mfcc

    def __call__(self, signal):
        if self.feature_extract_by == "torchaudio":
            mfcc = self.transforms(FloatTensor(signal))
            mfcc = mfcc.numpy()

        elif self.feature_extract_by == "librosa":
            mfcc = self.transforms(
                y=signal,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

        else:
            raise ValueError(
                "Unsupported library : {0}".format(self.feature_extract_by)
            )

        return mfcc


class FilterBank(object):
    """
    Create a fbank from a raw audio signal. This matches the input/output of Kaldi’s compute-fbank-feats

    Args:
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mels (int):  Number of mfc coefficients to retain. (Default: 80)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        frame_length: int = 20,
        frame_shift: int = 10,
    ) -> None:
        try:
            import torchaudio
        except ImportError:
            raise ImportError("Please install torchaudio `pip install torchaudio`")
        self.transforms = torchaudio.compliance.kaldi.fbank
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift

    def __call__(self, signal):
        return (
            self.transforms(
                Tensor(signal).unsqueeze(0),
                num_mel_bins=self.n_mels,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
            )
            .transpose(0, 1)
            .numpy()
        )
