"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import imgaug.augmenters as iaa
import numpy as np

from synthtiger.components.component import Component


class MotionBlur(Component):
    def __init__(self, k=(3, 7), angle=(0, 360)):
        super().__init__()
        self.k = k
        self.angle = angle

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        k = meta.get("k", np.random.randint(self.k[0], self.k[1] + 1))
        angle = meta.get("angle", np.random.uniform(self.angle[0], self.angle[1]))

        meta = {
            "k": k,
            "angle": angle,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        k = meta["k"]
        angle = meta["angle"]
        aug = iaa.MotionBlur(k=k, angle=angle)

        for layer in layers:
            image = layer.image.astype(np.uint8)
            image = aug(image=image).astype(np.float32)
            layer.image = image

        return meta
