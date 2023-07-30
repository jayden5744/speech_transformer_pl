import os
import re

from typing import Tuple, List

from .character import generate_character_labels, generate_character_script
from .subword import train_sentencepiece, sentence_to_subwords
from .grapheme import sentence_to_grapheme


def bracket_filter(sentence: str, mode: str ='phonetic') -> str:
    new_sentence = str()

    if mode == 'phonetic':
        flag = False

        for ch in sentence:
            if ch == '(' and flag is False:
                flag = True
                continue
            if ch == '(' and flag is True:
                flag = False
                continue
            if ch != ')' and flag is False:
                new_sentence += ch

    elif mode == 'spelling':
        flag = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ')' and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode : {0}".format(mode))

    return new_sentence


def special_filter(sentence: str, mode: str = 'phonetic', replace: str = None) -> str:
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'

        elif ch == '%':
            if mode == 'phonetic':
                new_sentence += replace
            elif mode == 'spelling':
                new_sentence += '%'

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence: str, mode: str, replace: str = None) -> str:
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)


def preprocess(dataset_path: str, mode: str = 'phonetic') -> Tuple[List[str], List[str]]:
    print('preprocess started..')

    audio_paths = list()
    transcripts = list()

    percent_files = {
        '087797': '퍼센트',
        '215401': '퍼센트',
        '284574': '퍼센트',
        '397184': '퍼센트',
        '501006': '프로',
        '502173': '프로',
        '542363': '프로',
        '581483': '퍼센트'
    }

    for folder in os.listdir(dataset_path):
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        if not folder.startswith('KsponSpeech'):
            continue
        path = os.path.join(dataset_path, folder)
        for idx, subfolder in enumerate(os.listdir(path)):
            path = os.path.join(dataset_path, folder, subfolder)

            for jdx, file in enumerate(os.listdir(path)):
                if file.endswith('.txt'):
                    with open(os.path.join(path, file), "r", encoding='cp949') as f:
                        raw_sentence = f.read()
                        if file[12:18] in percent_files.keys():
                            new_sentence = sentence_filter(raw_sentence, mode, percent_files[file[12:18]])
                        else:
                            new_sentence = sentence_filter(raw_sentence, mode=mode)

                    audio_paths.append(os.path.join(folder, subfolder, file))
                    transcripts.append(new_sentence)

                else:
                    continue

    return audio_paths, transcripts


def preprocess_kspon(
    dataset_path: str,
    output_unit: str,
    save_path: str,
    preprocess_mode: str,
    vocab_size: int
    ):
    assert output_unit in ["character", "subword", "grapheme"], "output_unit must be one of [character, subword, grapheme]"
    assert preprocess_mode in ["spelling", "phonetic"], "preprocess_mode must be one of [spelling, phonetic]"

    audio_paths, transcripts = preprocess(dataset_path, preprocess_mode)


    if output_unit == 'character':
        generate_character_labels(transcripts, save_path)
        generate_character_script(audio_paths, transcripts, save_path)

    elif output_unit == 'subword':
        train_sentencepiece(transcripts, save_path, vocab_size)
        sentence_to_subwords(audio_paths, transcripts, save_path)

    elif output_unit == 'grapheme':
        sentence_to_grapheme(audio_paths, transcripts, save_path)

    else:
        raise ValueError("Unsupported preprocess method : {0}".format(output_unit))
