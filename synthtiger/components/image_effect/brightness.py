"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component


class Brightness(Component):
    def __init__(self, beta=(-32, 32)):
        super().__init__()
        self.beta = beta

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        beta = meta.get("beta", np.random.randint(self.beta[0], self.beta[1] + 1))

        meta = {
            "beta": beta,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        beta = meta["beta"]

        for layer in layers:
            layer.image[..., :3] += beta
            layer.image[..., :3] = np.clip(layer.image[..., :3], 0, 255)

        return meta
