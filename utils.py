"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
import sys
from multiprocessing import Process

import numpy as np
import yaml
from PIL import Image


def run_process(func, args):
    proc = Process(target=func, args=args)
    proc.daemon = True
    proc.start()
    return proc


def read_template(path):
    path = os.path.abspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    sys.path.append(os.path.dirname(path))
    template = getattr(__import__(name), "Template")
    return template


def read_config(path):
    with open(path, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    return config


def write_image(path, image, quality=95):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg"]:
        image = Image.fromarray(image[..., :3].astype(np.uint8))
    if ext in [".png"]:
        image = Image.fromarray(image.astype(np.uint8))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, quality=quality)


def create_gt(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fp = open(path, "w", encoding="utf-8")
    return fp


def write_gt(fp, key, value):
    fp.write(f"{key}\t{value}\n")
