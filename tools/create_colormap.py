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
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.cluster
from PIL import Image


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


def write_cluster(fp, clusters):
    texts = []

    for center, std in clusters:
        center = ",".join(list(map(str, center)))
        std = str(std)
        texts.append(f"{center}\t{std}")

    text = "\t".join(texts)
    fp.write(f"{text}\n")


def get_cluster(path, k, rgb=False):
    clusters = []

    mode = "RGB" if rgb else "L"
    channel = 3 if rgb else 1
    image = Image.open(path).convert(mode)
    image = np.array(image, dtype=np.float32).reshape(-1, channel)

    centers, _ = scipy.cluster.vq.kmeans(image, k)
    if len(centers) != k:
        return None

    vecs, _ = scipy.cluster.vq.vq(image, centers)
    stds = [np.std(image[vecs == idx]) for idx in range(len(centers))]

    for center, std in zip(centers, stds):
        clusters.append((list(center), std))

    clusters = sorted(clusters)
    return clusters


def run(args):
    paths = search_files(args.input, exts=[".jpg", ".jpeg", ".png", ".bmp"])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_file = open(args.output, "w", encoding="utf-8")
    executor = ProcessPoolExecutor(max_workers=args.worker)
    count = 0

    for k in range(2, args.max_k + 1):
        futures = {}

        for path in paths:
            future = executor.submit(get_cluster, path, k, args.rgb)
            futures[future] = path

        for future in as_completed(futures):
            path = futures[future]

            try:
                clusters = future.result()
            except:
                print(f"{traceback.format_exc()} ({path})")
                continue

            if clusters is not None:
                write_cluster(output_file, clusters)
                count += 1
                print(f"Created colormap ({k} colors) ({path})")

    executor.shutdown()
    output_file.close()
    print(f"Created {count} colormaps")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rgb",
        action="store_true",
        default=False,
        help="Create rgb colormap instead of gray colormap.",
    )
    parser.add_argument(
        "--max_k",
        metavar="NUM",
        type=int,
        default=3,
        help="Maximum number of colors. [default: 3]",
    )
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
        help="Directory path containing image files.",
    )
    parser.add_argument(
        "output",
        metavar="OUTPUT",
        type=str,
        help="File path to save colormap.",
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
