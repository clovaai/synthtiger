"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component


class Opacity(Component):
    def __init__(self, opacity=(0, 1)):
        super().__init__()
        self.opacity = opacity

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        opacity = meta.get(
            "opacity", np.random.uniform(self.opacity[0], self.opacity[1])
        )

        meta = {
            "opacity": opacity,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        opacity = meta["opacity"]

        for layer in layers:
            layer.image[..., 3] *= opacity

        return meta
