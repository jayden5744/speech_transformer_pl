import os
import re
from typing import List, Tuple

from src.utils import logger

from .character import generate_character_labels, generate_character_script
from .grapheme import generate_grapheme_labels, generate_grapheme_script
from .subword import sentence_to_subwords, train_sentencepiece


def bracket_filter(sentence: str, mode: str = "phonetic") -> str:
    # bracket 괄호를 제거 & phonetic or spelling 중 하나만 택
    new_sentence = ""

    if mode == "phonetic":
        flag = False
        for ch in sentence:
            if ch == "(" and flag is False:
                flag = True

            elif ch == "(" and flag is True:
                flag = False

            elif ch != ")" and flag is False:
                new_sentence += ch

    elif mode == "spelling":
        flag = True
        for ch in sentence:
            if ch == "(":
                continue

            elif ch == ")" and flag is True:
                flag = False

            elif ch == ")" and flag is False:
                flag = True

            elif ch != ")" and flag is True:
                new_sentence += ch

    else:
        raise ValueError(f"Unsupported mode : {mode}")

    return new_sentence


def special_filter(sentence: str, mode: str = "phonetic", replace: str = None) -> str:
    # 특수문자 제거
    # 숫자가 포함되어 있는 부분 어떻게 읽는지
    SENTENCE_MARK = ["?", "!", "."]
    NOISE = ["o", "n", "u", "b", "l"]
    EXCEPT = ["/", "+", "*", "-", "@", "$", "^", "&", "[", "]", "=", ":", ";", ","]

    new_sentence = ""

    for idx, ch in enumerate(sentence):
        if (
            ch not in SENTENCE_MARK
            and idx + 1 < len(sentence)
            and ch in NOISE
            and sentence[idx + 1] == "/"
        ):
            # character가 sentence_mark에 포함되지 않고 idx가 마지막 character이고, ch가 노이즈이고, 마지막이 "/"이거 일때 무시
            continue

        if ch == "#":
            new_sentence += "샾"

        elif ch == "%":
            if mode == "phonetic":
                new_sentence += replace

            else:
                new_sentence += "%"

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r"\s\s+")
    new_sentence = re.sub(pattern, " ", new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence: str, mode: str, replace: str = None) -> str:
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)


def preprocess(
    dataset_path: str, mode: str = "phonetic"
) -> Tuple[List[str], List[str]]:
    logger.info("Preprocess started")

    audio_paths = []
    transcripts = []

    percent_files = {
        "087797": "퍼센트",
        "215401": "퍼센트",
        "284574": "퍼센트",
        "397184": "퍼센트",
        "501006": "프로",
        "502173": "프로",
        "542363": "프로",
        "581483": "퍼센트",
    }

    for folder in os.listdir(dataset_path):
        # folder : {KsponSpeech_01, KsponSpeech_02, ..., KsponSpeech_05}
        if not folder.startswith("KsponSpeech"):
            continue
        for idx, sub_folder in enumerate(
            os.listdir(os.path.join(dataset_path, folder))
        ):
            # sub_folder : KsponSpeech_0001, ....
            path = os.path.join(dataset_path, folder, sub_folder)
            for jdx, file in enumerate(os.listdir(path)):
                replace = None
                if file.endswith(".txt"):
                    with open(os.path.join(path, file), "r", encoding="cp949") as f:
                        raw_sentence = f.read()
                        if file[12:18] in percent_files.keys():
                            replace = percent_files[file[12:18]]

                        processed_sentences = sentence_filter(
                            raw_sentence, mode, replace
                        )

                    audio_paths.append(
                        os.path.join(folder, sub_folder, file[:-4] + ".pcm")
                    )  # Todo : file 부분 확장자 제외하는 지 확인 필요
                    transcripts.append(processed_sentences)

                else:
                    continue
    return audio_paths, transcripts


def prepross_kspon(
    dataset_path: str,
    output_unit: str,
    save_path: str,
    preprocess_mode: str,
    vocab_size: int,
):
    assert output_unit in [
        "character",
        "subword",
        "grapheme",
    ], "output_unit must be one of [character, subword, grapheme]"
    assert preprocess_mode in [
        "spelling",
        "phonetic",
    ], "preprocess_mode must be one of [spelling, phonetic]"

    audio_paths, transcripts = preprocess(dataset_path, preprocess_mode)

    # output unit 별로 처리
    if output_unit == "character":
        generate_character_labels(transcripts, save_path)
        generate_character_script(audio_paths, transcripts, save_path)

    elif output_unit == "subword":
        train_sentencepiece(transcripts, save_path, vocab_size)
        sentence_to_subwords(audio_paths, transcripts, save_path)

    elif output_unit == "grapheme":
        generate_grapheme_labels(transcripts, save_path)
        generate_grapheme_script(audio_paths, transcripts, save_path)

    else:
        raise ValueError(f"Unsupported preprocess method : {output_unit}")
