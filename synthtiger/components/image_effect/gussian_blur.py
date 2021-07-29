"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import imgaug.augmenters as iaa
import numpy as np

from synthtiger.components.component import Component


class GaussianBlur(Component):
    def __init__(self, sigma=(1, 2)):
        super().__init__()
        self.sigma = sigma

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        sigma = meta.get("sigma", np.random.randint(self.sigma[0], self.sigma[1] + 1))

        meta = {
            "sigma": sigma,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        sigma = meta["sigma"]
        aug = iaa.GaussianBlur(sigma=sigma)

        for layer in layers:
            image = layer.image.astype(np.uint8)
            image = aug(image=image).astype(np.float32)
            layer.image = image

        return meta
