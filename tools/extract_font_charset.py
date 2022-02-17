"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import argparse
import os
import pprint
import time
import traceback
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed

from fontTools.ttLib import TTFont
from pygame import freetype


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


def get_cmap(path):
    cmap = set()
    font = TTFont(path)

    for table in font["cmap"].tables:
        for code, _ in table.cmap.items():
            try:
                char = chr(code)
            except:
                continue

            cmap.add(char)

    return cmap


def get_glyphs(path):
    glyphs = {}
    cmap = get_cmap(path)

    freetype.init()
    font = freetype.Font(path)
    font.antialiased = True
    font.pad = True
    font.size = 72

    for char in cmap:
        category = unicodedata.category(char)
        if category.startswith("C"):
            continue

        try:
            glyph, _ = font.render_raw(char)
        except:
            continue

        if glyph not in glyphs:
            glyphs[glyph] = []
        glyphs[glyph].append(char)

    return glyphs


def get_charset(path):
    charset = set()
    glyphs = get_glyphs(path)
    threshold = 10

    for glyph, chars in glyphs.items():
        empty = sum(glyph) == 0

        if not empty and len(chars) > threshold:
            continue

        for char in chars:
            space = char.isspace()
            if (empty and not space) or (not empty and space):
                continue
            charset.add(char)

    return charset


def write_charset(path, charset):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    charset = sorted(charset)
    with open(path, "w", encoding="utf-8") as fp:
        for char in charset:
            fp.write(char)


def run(args):
    paths = search_files(args.input, exts=[".ttf", ".otf"])
    executor = ProcessPoolExecutor(max_workers=args.worker)
    futures = {}
    count = 0

    for path in paths:
        future = executor.submit(get_charset, path)
        futures[future] = path

    for future in as_completed(futures):
        path = futures[future]

        try:
            charset = future.result()
        except:
            print(f"{traceback.format_exc()} ({path})")
            continue

        output_path = f"{os.path.splitext(path)[0]}.txt"
        write_charset(output_path, charset)
        count += 1
        print(f"Extracted font charset ({len(charset)} chars) ({path})")

    executor.shutdown()
    print(f"Extracted {count} font charsets")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--worker",
        metavar="NUM",
        type=int,
        default=1,
        help="Number of workers. [default: 1]",
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        type=str,
        help="Directory path containing font files.",
    )
    args = parser.parse_args()

    pprint.pprint(vars(args))

    return args


def main():
    start_time = time.time()
    args = parse_args()
    run(args)
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")


if __name__ == "__main__":
    main()
