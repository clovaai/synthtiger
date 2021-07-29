"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os


def search_files(root, names=None, exts=None):
    paths = []

    for dir_path, _, file_names in os.walk(root):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            file_ext = os.path.splitext(file_name)[1]

            if names is not None and file_name not in names:
                continue
            if exts is not None and file_ext.lower() not in exts:
                continue

            paths.append(file_path)

    return paths


def read_charset(path):
    with open(path, "r", encoding="utf-8") as fp:
        charset = set(fp.read())
    return charset
