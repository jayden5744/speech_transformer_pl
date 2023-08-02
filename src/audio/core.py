import librosa
import numpy as np

from src.utils import logger


def load_audio(
    audio_path: str, del_silence: bool = False, extension: str = "pcm", sr: int = 16000
) -> np.ndarray:
    "Load Audio"
    try:
        if extension == "pcm":  # numpy
            signal = np.memmap(audio_path, dtype="h", mode="r").astype("float32")

            if del_silence:
                non_silence_indices = librosa.effects.split(
                    signal, top_db=30
                )  # top_db: 데시벨보다 낮은 부분을 제외하고 index 찾아주는 거
                signal = np.concatenate(
                    [signal[start:end] for start, end in non_silence_indices]
                )
            return signal

        elif extension in ["wav", "flac"]:  # librosa
            signal, _ = librosa.load(audio_path, sr=sr)
            return signal

    except ValueError:
        logger.debug(f"ValueError in {audio_path}")

    except RuntimeError:
        logger.debug(f"RuntimeError in {audio_path}")

    except IOError:
        logger.debug(f"IOError in {audio_path}")
