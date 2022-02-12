"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component


class Contrast(Component):
    def __init__(self, alpha=(0.5, 1.5)):
        super().__init__()
        self.alpha = alpha

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        alpha = meta.get("alpha", np.random.uniform(self.alpha[0], self.alpha[1]))

        meta = {
            "alpha": alpha,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        alpha = meta["alpha"]

        for layer in layers:
            layer.image[..., :3] = alpha * layer.image[..., :3] - 128 * (alpha - 1)
            layer.image[..., :3] = np.clip(layer.image[..., :3], 0, 255)

        return meta
