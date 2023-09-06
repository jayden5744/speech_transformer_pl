import logging
import random

import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
