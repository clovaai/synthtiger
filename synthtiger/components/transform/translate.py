"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component
from synthtiger.layers import Group


class Translate(Component):
    def __init__(self, offset=((-1, 1), (-1, 1))):
        super().__init__()
        self.offset = offset

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        offset = meta.get(
            "offset",
            (
                np.random.uniform(self.offset[0][0], self.offset[0][1]),
                np.random.uniform(self.offset[1][0], self.offset[1][1]),
            ),
        )

        meta = {
            "offset": offset,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        offset = meta["offset"]

        group = Group(layers)
        group.quad += group.size * offset

        return meta
