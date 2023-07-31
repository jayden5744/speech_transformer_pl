import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3.2", config_path="configs", config_name="pretrain")
def train(cfg: DictConfig) -> None:
    pass


if __name__ == "__main__":
    train()
