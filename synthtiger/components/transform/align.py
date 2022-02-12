"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component
from synthtiger.layers import Group


class Align(Component):
    def __init__(self, aligns=((-1, 1), (-1, 1))):
        super().__init__()
        self.aligns = aligns

        shapes = [(1, 2), (2, 2)]
        if np.array(self.aligns).shape not in shapes:
            raise TypeError("Shape of aligns must be (1,2) or (2,2)")

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        aligns = meta.get(
            "aligns",
            tuple(np.random.uniform(align[0], align[1]) for align in self.aligns),
        )

        meta = {
            "aligns": aligns,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        aligns = meta["aligns"]

        aligns = np.tile(aligns, 2)[:2]
        aligns = np.clip(aligns, -1, 1)
        aligns = (aligns + 1) / 2

        group = Group(layers)
        topleft = group.topleft
        size = group.size

        for layer in layers:
            layer.topleft = topleft
            layer.quad += (size - layer.size) * aligns

        return meta
