from typing import Dict

import lightning.pytorch as pl
import sentencepiece as spm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import Optimizer

from src.exceptions import ParameterError
from src.model.transformer import SpeechTransformer
from src.preprocess import load_vocab


class SpeechTransformerModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab = self.get_vocab()
        self.model = self.build_model()

        self.pad_id = self.vocab.pad_id()

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id,
            label_smoothing=self.cfg.trainer.label_smoothing_value,
        )

    def _shared_eval_step(self, batch, batch_idx: int) -> Tensor:
        # validation step과 test step의 공통으로 사용되는 부분
        inputs, targets, input_lengths, target_lengths = batch
        target_lengths = torch.as_tensor(target_lengths)

        outputs, encoder_output_lengths, encoder_log_probs = self.model(
            inputs, input_lengths, targets, target_lengths
        )
        return self.calculate_loss(outputs, targets)

    def training_step(self, batch, batch_idx: int) -> Dict[str, Tensor]:
        inputs, targets, input_lengths, target_lengths = batch
        target_lengths = torch.as_tensor(target_lengths)

        outputs, encoder_output_lengths, encoder_log_probs = self.model(
            inputs, input_lengths, targets, target_lengths
        )
        loss = self.calculate_loss(outputs, targets)

        metrics = {"loss": loss}
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx: int) -> Dict[str, Tensor]:
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def on_validation_epoch_end(self):
        # validation 1 epoch 끝나고 나서 수행하게 될 로직
        pass

    def calculate_loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        if self.device.type == "mps":
            # mps float64를 처리할 수 없음
            # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
            outputs = outputs.to(device="cpu")
            targets = targets.to(device="cpu")

        return self.criterion(
            outputs.contiguous().view(-1, outputs.size(-1)),
            targets[:, 1:].contiguous().view(-1),
        )

    def configure_optimizers(self) -> Optimizer:
        optimizer_type = self.cfg.trainer.optimizer
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.trainer.learning_rate,
                betas=(self.cfg.trainer.optimizer_b1, self.cfg.trainer.optimizer_b2),
                eps=self.cfg.trainer.optimizer_e,
                weight_decay=self.cfg.trainer.weight_decay,
            )

        elif optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.trainer.learning_rate,
                betas=(self.cfg.trainer.optimizer_b1, self.cfg.trainer.optimizer_b2),
                eps=self.cfg.trainer.optimizer_e,
                weight_decay=self.cfg.trainer.weight_decay,
            )

        else:
            raise ValueError("trainer param `optimizer` must be one of [Adam, AdamW].")
        return optimizer

    def get_vocab(self) -> spm.SentencePieceProcessor:
        return load_vocab(vocab_path=self.cfg.trainer.vocab_path)

    def build_model(self) -> nn.Module:
        if self.cfg.audio.transform_method.lower() == "spect":
            if self.cfg.audio.feature_extract_by == "kaldi":
                input_size = 257

            else:
                input_size = (self.cfg.audio.frame_length << 3) + 1

        else:
            input_size = self.cfg.audio.n_mels

        if self.cfg.model.dropout < 0.0:
            raise ParameterError("dropout probability shoul be positive")

        if input_size < 0:
            raise ParameterError("`input size` should be greater than 0")

        if (
            self.cfg.model.num_encoder_layers < 0
            or self.cfg.model.num_decoder_layers < 0
        ):
            raise ParameterError("`num_layers` should be greater than 0")

        return SpeechTransformer(
            input_dim=input_size,
            num_classes=len(self.vocab),
            extractor=self.cfg.model.extractor,
            num_encoder_layers=self.cfg.model.num_encoder_layers,
            num_decoder_layers=self.cfg.model.num_decoder_layers,
            dropout_p=self.cfg.model.dropout,
            d_model=self.cfg.model.d_model,
            d_ff=self.cfg.model.d_ff,
            pad_id=self.vocab.pad_id(),
            sos_id=self.vocab.bos_id(),
            eos_id=self.vocab.eos_id(),
            num_heads=self.cfg.model.num_heads,
            max_length=self.cfg.model.max_len,
        )
