"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os

import numpy as np
from PIL import Image

from synthtiger import components, layers, templates


class Multiline(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.count = config.get("count", 100)
        self.corpus = components.BaseCorpus(**config.get("corpus", {}))
        self.font = components.BaseFont(**config.get("font", {}))
        self.color = components.RGB(**config.get("color", {}))
        self.layout = components.FlowLayout(**config.get("layout", {}))

    def generate(self):
        texts = [self.corpus.data(self.corpus.sample()) for _ in range(self.count)]
        fonts = [self.font.sample() for _ in range(self.count)]
        color = self.color.data(self.color.sample())

        text_group = layers.Group(
            [
                layers.TextLayer(text, color=color, **font)
                for text, font in zip(texts, fonts)
            ]
        )
        self.layout.apply(text_group)

        bg_layer = layers.RectLayer(text_group.size, (255, 255, 255, 255))
        bg_layer.topleft = text_group.topleft

        image = (text_group + bg_layer).output()
        label = " ".join(texts)

        data = {
            "image": image,
            "label": label,
        }

        return data

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)
        gt_path = os.path.join(root, "gt.txt")
        self.gt_file = open(gt_path, "w", encoding="utf-8")

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]

        shard = str(idx // 10000)
        image_key = os.path.join("images", shard, f"{idx}.jpg")
        image_path = os.path.join(root, image_key)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_path, quality=95)

        self.gt_file.write(f"{image_key}\t{label}\n")

    def end_save(self, root):
        self.gt_file.close()
