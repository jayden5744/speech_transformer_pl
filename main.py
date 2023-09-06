import os

import hydra
from hydra.utils import get_original_cwd
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from src.data_modules import STTDataModule
from src.preprocess import prepross_kspon
from src.train import SpeechTransformerModule
from src.utils import set_seed


@hydra.main(version_base="1.3.2", config_path="configs", config_name="preprocess")
def preprocess(cfg: DictConfig) -> None:
    prepross_kspon(
        dataset_path=cfg.preprocess.dataset_path,
        output_unit=cfg.preprocess.output_unit,
        save_path=cfg.preprocess.save_path,
        preprocess_mode=cfg.preprocess.preprocess_mode,
        vocab_size=cfg.preprocess.vocab_size,
    )


@hydra.main(veråsion_base="1.3.2", config_path="configs", config_name="pretrain")
def train(cfg: DictConfig) -> None:
    callback_lst = []
    set_seed(cfg.trainer.seed)
    module = SpeechTransformerModule(cfg=cfg)
    data_module = STTDataModule(
        arg=cfg, vocab=module.vocab, batch_size=cfg.trainer.batch_size
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(get_original_cwd(), f"./SavedModel/kspon"),
        filename="kspon",
        save_top_k=5,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.trainer.early_stopping,
        verbose=False,
        mode="min",
    )
    callback_lst += [checkpoint_callback, early_stop_callback]

    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=cfg.trainer.epochs,
        callbacks=callback_lst,
        strategy="ddp_find_unused_parameters_true",  # GPU 여러장 쓸 때
    )
    trainer.fit(model=module, datamodule=data_module)


if __name__ == "__main__":
    preprocess()
