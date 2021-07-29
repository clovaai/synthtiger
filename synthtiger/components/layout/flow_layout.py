"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component


class FlowLayout(Component):
    def __init__(self, space=(0, 0), vertical=False):
        super().__init__()
        self.space = space
        self.vertical = vertical

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        space = meta.get("space", np.random.randint(self.space[0], self.space[1] + 1))
        vertical = meta.get("vertical", self.vertical)

        meta = {
            "space": space,
            "vertical": vertical,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        space = meta["space"]
        vertical = meta["vertical"]

        for layer in layers:
            layer.center = (0, 0)

        if vertical:
            for idx in range(1, len(layers)):
                layers[idx].top = layers[idx - 1].bottom + space
        else:
            for idx in range(1, len(layers)):
                layers[idx].left = layers[idx - 1].right + space

        return meta
