"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np

from synthtiger.components.component import Component
from synthtiger.layers import Group


class Perspective(Component):
    def __init__(
        self, pxs=None, percents=None, aligns=((-1, 1), (-1, 1), (-1, 1), (-1, 1))
    ):
        super().__init__()
        self.pxs = pxs
        self.percents = percents
        self.aligns = aligns

        shapes = [(1, 2), (2, 2), (3, 2), (4, 2)]
        if self.pxs is not None and np.array(self.pxs).shape not in shapes:
            raise TypeError("Shape of pxs must be (1,2), (2,2), (3,2) or (4,2)")
        if self.percents is not None and np.array(self.percents).shape not in shapes:
            raise TypeError("Shape of percents must be (1,2), (2,2), (3,2) or (4,2)")
        if np.array(self.aligns).shape not in shapes:
            raise TypeError("Shape of aligns must be (1,2), (2,2), (3,2) or (4,2)")

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
        aligns = meta.get(
            "aligns",
            tuple(np.random.uniform(align[0], align[1]) for align in self.aligns),
        )

        meta = {
            "pxs": pxs,
            "percents": percents,
            "aligns": aligns,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        pxs = meta["pxs"]
        percents = meta["percents"]
        aligns = meta["aligns"]

        aligns = np.tile(aligns, 4)[:4]
        if pxs is not None:
            pxs = np.tile(pxs, 4)[:4]
        if percents is not None:
            percents = np.tile(percents, 4)[:4]

        group = Group(layers)
        sizes = np.tile(group.size, 2)
        new_sizes = np.tile(group.size, 2)

        if pxs is not None:
            new_sizes += pxs
        elif percents is not None:
            new_sizes *= percents

        values = (sizes - new_sizes) / 2
        aligns *= np.abs(values)
        offsets = [
            [values[0] + aligns[0], values[3] + aligns[3]],
            [-values[0] + aligns[0], values[1] + aligns[1]],
            [-values[2] + aligns[2], -values[1] + aligns[1]],
            [values[2] + aligns[2], -values[3] + aligns[3]],
        ]

        origin = group.quad
        quad = np.array(origin + offsets, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(origin, quad)

        for layer in layers:
            quad = np.append(layer.quad, np.ones((4, 1)), axis=-1).dot(matrix.T)
            layer.quad = quad[..., :2] / quad[..., 2, np.newaxis]

        return meta
