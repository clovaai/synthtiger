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
    data = {}

    with open(os.path.join(root, _VERT_ORIENT_PATH), "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue

            unicode_range, value = line.split(";")
            unicode_range = unicode_range.strip()
            value = value.strip()

            unicodes = unicode_range.split("..")
            unicodes = [int(unicode, base=16) for unicode in unicodes]

            if len(unicodes) == 1:
                data[unicodes[0]] = value

            elif len(unicodes) == 2:
                for unicode in range(unicodes[0], unicodes[1] + 1):
                    data[unicode] = value

    return data


# https://unicode.org/Public/vertical
_VERT_ORIENT_PATH = "VerticalOrientation-17.txt"
_VERT_ORIENT = _read_vert_orient()

# http://www.unicode.org/reports/tr50
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
    unicode = ord(char)
    return _VERT_ORIENT[unicode] if unicode in _VERT_ORIENT else "R"


def vert_rot_flip(char):
    unicode = ord(char)
    return unicode in _VERT_ROT_FLIP


def vert_right_flip(char):
    unicode = ord(char)
    return unicode in _VERT_RIGHT_FLIP


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
