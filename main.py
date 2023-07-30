
import hydra
from omegaconf import DictConfig

from src.preprocess import preprocess_kspon


@hydra.main(version_base="1.3.2", config_path="configs", config_name="config")
def preprocess(cfg: DictConfig) -> None:
    preprocess_kspon(
        dataset_path=cfg.preprocess.dataset_path,
        output_unit=cfg.preprocess.output_unit,
        save_path=cfg.preprocess.save_path,
        preprocess_mode=cfg.preprocess.preprocess_mode,
        vocab_size=cfg.preprocess.vocab_size
    )



@hydra.main(version_base="1.3.2", config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    pass



if __name__ == "__main__":
    preprocess()