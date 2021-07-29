"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import imgaug.augmenters as iaa
import numpy as np

from synthtiger.components.component import Component


class ImageRotate(Component):
    def __init__(self, angle=(-45, 45), ccw=0, mode="constant"):
        super().__init__()
        self.angle = angle
        self.ccw = ccw
        self.mode = mode

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        angle = meta.get("angle", np.random.uniform(self.angle[0], self.angle[1]))
        ccw = meta.get("ccw", np.random.rand() < self.ccw)
        mode = meta.get("mode", self.mode)

        meta = {
            "angle": angle,
            "ccw": ccw,
            "mode": mode,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        angle = meta["angle"] * (-1 if meta["ccw"] else 1)
        mode = meta["mode"]
        aug = iaa.Rotate(rotate=angle, mode=mode)

        for layer in layers:
            image = layer.image.astype(np.uint8)
            image = aug(image=image).astype(np.float32)
            layer.image = image

        return meta
