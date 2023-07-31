import os
from typing import Dict, List, Tuple

import pandas as pd

from src.utils import logger


def generate_character_labels(transcripts: List[str], save_path: str):
    logger.info("Create character labels started")
    label_dict = {}
    os.makedirs(save_path, exist_ok=True)

    for transcript in transcripts:
        for ch in transcript:
            if ch not in label_dict.keys():
                label_dict[ch] = 1

            else:
                label_dict[ch] += 1

    label_freq, label_list = zip(
        *sorted(zip(label_dict.values(), label_dict.keys()), reverse=True)
    )
    label = {"id": [0, 1, 2], "char": ["<pad>", "<sos>", "<eos>"], "freq": [0, 0, 0]}

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label["id"].append(idx + 3)
        label["char"].append(ch)
        label["freq"].append(freq)

    label_df = pd.DataFrame(label)
    label_df.to_csv(
        os.path.join(save_path, "aihub_labels.csv"), encoding="utf-8", index=False
    )


def load_label(label_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    char2id = {}
    id2char = {}

    ch_labels = pd.read_csv(label_path, encoding="utf-8-sig")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]

    for id_, char in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence2id(sentence: str, char2id: Dict[str, str]) -> str:
    id_lst = []
    for ch in sentence:
        try:
            id_lst.append(str(char2id[ch]))
        except KeyError:
            continue

    return " ".join(id_lst)


def generate_character_script(
    audio_paths: List[str], transcripts: List[str], save_path: str
) -> None:
    logger.info("Create script started")
    char2id, id2char = load_label(os.path.join(save_path, "aihub_labels.csv"))

    with open(os.path.join(save_path, "transcripts.txt"), "w") as f:
        for audio_path, transcript in zip(audio_paths, transcripts):
            char_id_transcript = sentence2id(transcript, char2id)
            f.write(f"{audio_path}\t{transcript}\t{char_id_transcript}\n")
