"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import imgaug.augmenters as iaa
import numpy as np

from synthtiger.components.component import Component


class ElasticDistortion(Component):
    def __init__(self, alpha=(10, 15), sigma=(3, 3)):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        alpha = meta.get("alpha", np.random.uniform(self.alpha[0], self.alpha[1]))
        sigma = meta.get("sigma", np.random.uniform(self.sigma[0], self.sigma[1]))

        meta = {
            "alpha": alpha,
            "sigma": sigma,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        alpha = meta["alpha"]
        sigma = meta["sigma"]
        aug = iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest")

        for layer in layers:
            image = layer.image.astype(np.uint8)
            image = aug(image=image).astype(np.float32)
            layer.image = image

        return meta
