"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np

from synthtiger.components.component import Component
from synthtiger.layers import Group


class Skew(Component):
    def __init__(self, weights=(1, 1), angle=(-30, 30), ccw=0):
        super().__init__()
        self.weights = weights
        self.angle = angle
        self.ccw = ccw
        self._probs = np.array(self.weights) / sum(self.weights)

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        axis = meta.get("axis", np.random.choice(2, p=self._probs))
        angle = meta.get("angle", np.random.uniform(self.angle[0], self.angle[1]))
        ccw = meta.get("ccw", np.random.rand() < self.ccw)

        meta = {
            "axis": axis,
            "angle": angle,
            "ccw": ccw,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        axis = meta["axis"]
        angle = meta["angle"] * (1 if meta["ccw"] else -1)

        group = Group(layers)
        values = [0, 0]
        values[axis] = group.size[::-1][axis] * np.tan(np.radians(angle)) / 2
        offsets = [
            [values[0], -values[1]],
            [values[0], values[1]],
            [-values[0], values[1]],
            [-values[0], -values[1]],
        ]

        origin = group.quad
        quad = np.array(origin + offsets, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(origin, quad)

        for layer in layers:
            quad = np.append(layer.quad, np.ones((4, 1)), axis=-1).dot(matrix.T)
            layer.quad = (quad / quad[..., 2, np.newaxis])[..., :2]

        return meta
