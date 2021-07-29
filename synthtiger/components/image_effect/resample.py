"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import imgaug.augmenters as iaa
import numpy as np

from synthtiger.components.component import Component


class Resample(Component):
    def __init__(self, size=(0.3, 0.7)):
        super().__init__()
        self.size = size

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        size = meta.get("size", np.random.uniform(self.size[0], self.size[1]))

        meta = {
            "size": size,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        size = meta["size"]
        aug = iaa.KeepSizeByResize(
            iaa.Resize(size=size, interpolation=["nearest", "linear", "area", "cubic"]),
            interpolation=["nearest", "linear", "area", "cubic"],
        )

        for layer in layers:
            image = layer.image.astype(np.uint8)
            image = aug(image=image).astype(np.float32)
            layer.image = image

        return meta
