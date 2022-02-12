"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np

from synthtiger.components.component import Component
from synthtiger.layers import Group


class Trapezoidate(Component):
    def __init__(self, weights=(1, 1, 1, 1), px=None, percent=None, align=(-1, 1)):
        super().__init__()
        self.weights = weights
        self.px = px
        self.percent = percent
        self.align = align
        self._probs = np.array(self.weights) / sum(self.weights)

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        side = meta.get("side", np.random.choice(4, p=self._probs))
        px = meta.get(
            "px",
            np.random.randint(self.px[0], self.px[1] + 1)
            if self.px is not None
            else None,
        )
        percent = meta.get(
            "percent",
            np.random.uniform(self.percent[0], self.percent[1])
            if self.percent is not None
            else None,
        )
        align = meta.get("align", np.random.uniform(self.align[0], self.align[1]))

        meta = {
            "side": side,
            "px": px,
            "percent": percent,
            "align": align,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        side = meta["side"]
        px = meta["px"]
        percent = meta["percent"]
        align = meta["align"]

        pxs = [0, 0, 0, 0]
        percents = [1, 1, 1, 1]
        aligns = [0, 0, 0, 0]

        aligns[side] = align
        if px is not None:
            pxs[side] = px
        if percent is not None:
            percents[side] = percent

        group = Group(layers)
        sizes = np.tile(group.size, 2)
        new_sizes = np.tile(group.size, 2)

        if px is not None:
            new_sizes += pxs
        elif percent is not None:
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
