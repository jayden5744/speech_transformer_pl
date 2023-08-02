import os
import random
from typing import List, Tuple

from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.audio.parser import SpectrogramParser
from src.utils import logger


def load_dataset(script_path: str) -> Tuple[List[str], List[str]]:
    audio_paths = []
    transcripts = []

    with open(script_path, "r") as f:
        for line in f.readlines():
            audio_path, _, transcript = line.split("\t")
            transcript = transcript.replace("\n", "")

            audio_paths.append(audio_path)
            transcripts.append(transcript)
    return audio_paths, transcripts


class SpectrogramDataset(Dataset, SpectrogramParser):
    """
    Dataset for feature & transcript matching

    Args:
        audio_paths (list): list of audio path
        transcripts (list): list of transcript
        sos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        spec_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        config (DictConfig): set of configurations
        dataset_path (str): path of dataset
    """

    def __init__(
        self,
        audio_paths: List[str],  # list of audio paths
        transcripts: List[str],  # list of transcript paths
        sos_id: int,  # identification of start of sequence token
        eos_id: int,  # identification of end of sequence token
        config: DictConfig,  # set of arguments
        spec_augment: bool = False,  # flag indication whether to use spec-augmentation of not
        dataset_path: str = None,  # path of dataset,
        audio_extension: str = "pcm",  # audio extension
    ) -> None:
        super(SpectrogramDataset, self).__init__(
            feature_extract_by=config.audio.feature_extract_by,
            sample_rate=config.audio.sample_rate,
            n_mels=config.audio.n_mels,
            frame_length=config.audio.frame_length,
            frame_shift=config.audio.frame_shift,
            del_silence=config.audio.del_silence,
            input_reverse=config.audio.input_reverse,
            normalize=config.audio.normalize,
            freq_mask_para=config.audio.freq_mask_para,
            time_mask_num=config.audio.time_mask_num,
            freq_mask_num=config.audio.freq_mask_num,
            sos_id=sos_id,
            eos_id=eos_id,
            dataset_path=dataset_path,
            transform_method=config.audio.transform_method,
            audio_extension=audio_extension,
        )
        self.audio_paths = list(audio_paths)
        self.transcripts = list(transcripts)
        self.augment_methods = [self.VANILLA] * len(self.audio_paths)
        self.dataset_size = len(self.audio_paths)
        self._augment(spec_augment)
        self.shuffle()

    def __getitem__(self, index):
        feature = self.parse_audio(
            os.path.join(self.dataset_path, self.audio_paths[index]),
            self.augment_methods[index],
        )

        if feature is None:
            return None, None

        transcript = self.parse_transcript(self.transcripts[index])
        return feature, transcript

    def parse_transcript(self, transcript: str) -> List[int]:
        tokens = transcript.split(" ")
        return [self.sos_id] + [int(token) for token in tokens] + [self.eos_id]

    def _augment(self, is_spec_augment: bool) -> None:
        if is_spec_augment:
            logger.info("Applying Spec Augmentation")

            for idx in range(self.dataset_size):
                self.augment_methods.append(self.SPEC_AUGMENT)
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])

    def shuffle(self):
        """Shuffle dataset"""
        tmp = list(zip(self.audio_paths, self.transcripts, self.augment_methods))
        random.shuffle(tmp)
        self.audio_paths, self.transcripts, self.augment_methods = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)
