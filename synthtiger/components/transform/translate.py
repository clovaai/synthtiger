"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component
from synthtiger.layers import Group


class Translate(Component):
    def __init__(self, pxs=None, percents=None):
        super().__init__()
        self.pxs = pxs
        self.percents = percents

        shapes = [(1, 2), (2, 2)]
        if self.pxs is not None and np.array(self.pxs).shape not in shapes:
            raise TypeError("Shape of pxs must be (1,2) or (2,2)")
        if self.percents is not None and np.array(self.percents).shape not in shapes:
            raise TypeError("Shape of percents must be (1,2) or (2,2)")

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        pxs = meta.get(
            "pxs",
            tuple(np.random.randint(px[0], px[1] + 1) for px in self.pxs)
            if self.pxs is not None
            else None,
        )
        percents = meta.get(
            "percents",
            tuple(
                np.random.uniform(percent[0], percent[1]) for percent in self.percents
            )
            if self.percents is not None
            else None,
        )

        meta = {
            "pxs": pxs,
            "percents": percents,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        pxs = meta["pxs"]
        percents = meta["percents"]

        if pxs is not None:
            pxs = np.tile(pxs, 2)[:2]
        if percents is not None:
            percents = np.tile(percents, 2)[:2]

        group = Group(layers)

        if pxs is not None:
            group.quad += pxs
        elif percents is not None:
            group.quad += group.size * percents

        return meta
