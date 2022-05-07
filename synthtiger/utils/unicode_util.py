"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
import unicodedata

import arabic_reshaper
import bidi.algorithm
import regex


def _read_vert_orient():
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "VerticalOrientation.txt")
    data = {}

    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = regex.sub("#.*", "", line).strip()
            if line == "":
                continue

            code_range, value = line.split(";")
            code_range = code_range.strip()
            value = value.strip()

            codes = code_range.split("..")
            codes = [int(code, base=16) for code in codes]

            if len(codes) == 1:
                data[codes[0]] = value
            if len(codes) == 2:
                for code in range(codes[0], codes[1] + 1):
                    data[code] = value

    return data


# vertical orientation
# http://www.unicode.org/reports/tr50
_VERT_ORIENT = _read_vert_orient()
_VERT_ROT_FLIP = [0x301C, 0x301E, 0x3030, 0x30FC, 0xFF5E]
_VERT_RIGHT_FLIP = [
    0x3001,
    0x3002,
    0x309B,
    0x309C,
    0xFE50,
    0xFE51,
    0xFE52,
    0xFF0C,
    0xFF0E,
]


def vert_orient(char):
    code = ord(char)
    return _VERT_ORIENT[code] if code in _VERT_ORIENT else "R"


def vert_rot_flip(char):
    code = ord(char)
    return code in _VERT_ROT_FLIP


def vert_right_flip(char):
    code = ord(char)
    return code in _VERT_RIGHT_FLIP


def to_fullwidth(text):
    chars = []

    for char in text:
        name = unicodedata.name(char)

        if "HALFWIDTH " in name:
            try:
                char = unicodedata.lookup(name.replace("HALFWIDTH ", ""))
            except:
                pass

        elif "FULLWIDTH " not in name:
            try:
                char = unicodedata.lookup(f"FULLWIDTH {name}")
            except:
                pass

        chars.append(char)

    text = "".join(chars)
    return text


def split_text(text, reorder=False, groups=None):
    if groups is None:
        groups = []

    text = reshape_text(text)
    groups = [reshape_text(group) for group in groups]
    if reorder:
        text = reorder_text(text)
        groups = [reorder_text(group) for group in groups]

    groups = set(groups)
    tokens = [text]
    chars = []

    if len(groups) > 0:
        pattern = [regex.escape(group) for group in groups]
        pattern = "({})".format("|".join(pattern))
        tokens = regex.split(pattern, text)
        tokens = list(filter(len, tokens))

    for token in tokens:
        if token in groups:
            chars.append(token)
        else:
            chars.extend(regex.findall(r"\X", token))

    return chars


def reshape_text(text):
    reshaper = arabic_reshaper.ArabicReshaper(
        {"use_unshaped_instead_of_isolated": True}
    )
    text = reshaper.reshape(text)
    return text


def reorder_text(text):
    text = bidi.algorithm.get_display(text)
    return text
