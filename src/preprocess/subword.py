import os
import os.path as osp
import shutil
from typing import List

import sentencepiece as spm

from src.utils import logger


def exist_file(path: str) -> bool:
    if osp.exists(path):
        return True
    return False


def train_sentencepiece(
    transcripts: List[str], save_path: str, vocab_size: int = 5000
) -> None:
    logger.info("generate_sentencepiece")
    model_prefix = "kspon_sentencepiece"

    os.makedirs(save_path, exist_ok=True)

    with open("sentencepiece_input.txt", "w", encoding="utf-8") as f:
        for transcript in transcripts:
            f.write(f"{transcript}\n")

    model_path = osp.join(save_path, model_prefix + ".model")
    vocab_path = osp.join(save_path, model_prefix + ".vocab")
    if not exist_file(model_path) and not exist_file(vocab_path):
        spm.SentencePieceTrainer.Train(
            f"--input=sentencepiece_input.txt "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} "
            f"--model_type=bpe "
            f"--pad_id=0 "
            f"--bos_id=1 "
            f"--eos_id=2 "
            f"--unk_id=3 "
        )
        shutil.move(model_prefix + ".model", model_path)
        shutil.move(model_prefix + ".vocab", vocab_path)


def sentence_to_subwords(
    audio_paths: List[str], transcripts: List[str], save_path: str
) -> None:
    logger.info("sentence to subwords")

    sp = spm.SentencePieceProcessor()
    model_file = osp.join(save_path, "kspon_sentencepiece.model")
    sp.load(model_file)

    with open(f"{save_path}/transcripts.txt", "w") as f:
        for audio_path, transcript in zip(audio_paths, transcripts):
            subword_transcript = " ".join(sp.EncodeAsPieces(transcript))
            subword_id_transcript = " ".join(
                [str(item) for item in sp.EncodeAsIds(transcript)]
            )
            f.write(f"{audio_path}\t{subword_transcript}\t{subword_id_transcript}\n")
