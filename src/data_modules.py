import multiprocessing
from typing import List, Tuple

import lightning.pytorch as pl
import sentencepiece as spm
import torch
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler

from .dataset import split_dataset


class CustomCollator(object):
    def __init__(self, pad_id: int = 0) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: Tuple[Tensor, List[int]]):
        """_summary_

        Args:
            batch (Tuple[Tensor, List[int]]): feature(음성), transcript(text) 의 Tuple

        Returns:
            seqs(Tensor) : [bs, max_seq_size(제일 긴 음성 샘플 길이), feature_size]
            targets(Tensor) : [bs, max_target_size(제일 긴 문장 길이)]
            seq_lengths(IntTensor): 음성 feature의 길이를 담은 Tensor
            target_lengths(List[int]): 문장 길이를 담은 List
        """

        def seq_length_(p):
            return len(p[0])

        def target_length_(p):
            return len(p[1])

        sorted_batch = sorted(
            batch, key=lambda x: x[0].size(), reverse=True
        )  # 음성길이 기준으로 긴것부터 정렬
        # sorted_batch = [(feature1, script1), (feature2, script2) ....]
        seq_lengths = [
            len(s[0]) for s in sorted_batch
        ]  # 음성 [feature1_length, feature2_length, ....]
        target_lengths = [
            len(s[1]) - 1 for s in sorted_batch
        ]  # 텍스트 [script1_length, script1_length, ....]

        max_seq_sample = max(sorted_batch, key=seq_length_)[
            0
        ]  # 제일 긴 음성 샘플(max_length_feature)
        max_target_sample = max(sorted_batch, key=target_length_)[
            1
        ]  # 제일 긴 문장 샘플 (max_length_script)

        max_seq_size = max_seq_sample.size(0)
        max_target_size = len(max_target_sample)

        feature_size = max_seq_sample.size(1)
        batch_size = len(batch)

        seqs = torch.zeros(batch_size, max_seq_size, feature_size)
        targets = torch.zeros(batch_size, max_target_size).to(torch.long)
        targets.fill_(self.pad_id)

        for x in range(batch_size):
            sample = batch[x]
            tensor = sample[0]  # 음성
            target = sample[1]  # 텍스트
            sample_length = tensor.size(0)

            seqs[x].narrow(0, 0, sample_length).copy_(tensor)
            targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

        seq_lengths = torch.IntTensor(seq_lengths)
        return seqs, targets, seq_lengths, target_lengths


class STTDataModule(pl.LightningDataModule):
    def __init__(
        self, arg: DictConfig, vocab: spm.SentencePieceProcessor, batch_size: int
    ) -> None:
        super().__init__()
        self.arg = arg
        self.vocab = vocab
        self.pad_id = self.vocab.pad_id()
        self.batch_size = batch_size
        self.train_dataset, self.valid_dataset = None, None

    def prepare_data(self) -> None:
        # 데이터를 다운로드, split 하거나 기타 등등
        # only called on 1 GPU/TPU in distributed
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        self.train_dataset, self.valid_dataset = split_dataset(
            config=self.arg,
            transcript_path=self.arg.trainer.transcripts_path,
            vocab=self.vocab,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        custom_collator = CustomCollator(self.pad_id)
        train_sampler = RandomSampler(self.train_dataset)
        return DataLoader(
            dataset=self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
            collate_fn=custom_collator,
            num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        custom_collator = CustomCollator(self.pad_id)
        valid_sampler = RandomSampler(self.valid_dataset)
        return DataLoader(
            dataset=self.valid_dataset,
            sampler=valid_sampler,
            batch_size=self.batch_size,
            collate_fn=custom_collator,
            num_workers=multiprocessing.cpu_count(),
            shuffle=False,  # validation에서는 shuffle 하지 않는 것은 권장함
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()

    def teardown(self, stage: str) -> None:
        # clean up after fit or test
        # called on every process in DDP
        # setup 정반대
        return super().teardown(stage)
