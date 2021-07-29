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
    def __init__(self, weights=(1, 1, 1, 1), scale=(0.5, 1.5), align=(-1, 1)):
        super().__init__()
        self.weights = weights
        self.scale = scale
        self.align = align

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        probs = np.array(self.weights) / sum(self.weights)
        side = meta.get("side", np.random.choice(4, p=probs))
        scale = meta.get("scale", np.random.uniform(self.scale[0], self.scale[1]))
        align = meta.get("align", np.random.uniform(self.align[0], self.align[1]))

        meta = {
            "side": side,
            "scale": scale,
            "align": align,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        side = meta["side"]
        scale = meta["scale"]
        align = meta["align"]

        group = Group(layers)
        scales = [1, 1, 1, 1]
        aligns = [0, 0, 0, 0]
        scales[side] = scale
        aligns[side] = align
        sizes = np.repeat(group.size, 2)
        values = (sizes - sizes * scales) / 2
        aligns *= np.abs(values)
        offsets = [
            [values[0] + aligns[0], values[2] + aligns[2]],
            [-values[0] + aligns[0], values[3] + aligns[3]],
            [-values[1] + aligns[1], -values[3] + aligns[3]],
            [values[1] + aligns[1], -values[2] + aligns[2]],
        ]

        origin = group.quad
        quad = np.array(origin + offsets, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(origin, quad)

        for layer in layers:
            quad = np.append(layer.quad, np.ones((4, 1)), axis=-1).dot(matrix.T)
            layer.quad = quad[..., :2] / quad[..., 2, np.newaxis]

        return meta
