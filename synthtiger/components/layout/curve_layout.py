"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np

from synthtiger.components.component import Component


class CurveLayout(Component):
    def __init__(
        self, curve=(10, 20), space=(0, 0), convex=0.5, upward=0.5, vertical=False
    ):
        super().__init__()
        self.curve = curve
        self.space = space
        self.convex = convex
        self.upward = upward
        self.vertical = vertical

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        curve = meta.get("curve", np.random.randint(self.curve[0], self.curve[1] + 1))
        space = meta.get("space", np.random.randint(self.space[0], self.space[1] + 1))
        convex = meta.get("convex", np.random.rand() < self.convex)
        upward = meta.get("upward", np.random.rand() < self.upward)
        vertical = meta.get("vertical", self.vertical)

        meta = {
            "curve": curve,
            "space": space,
            "convex": convex,
            "upward": upward,
            "vertical": vertical,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        curve = meta["curve"]
        space = meta["space"]
        convex = meta["convex"]
        upward = meta["upward"]
        vertical = meta["vertical"]

        orientation = -1 if convex else 1
        size = np.max([layer.size for layer in layers], axis=0)
        mid = (len(layers) - 1) / 2

        if vertical:
            w = curve * orientation
            h = (size[1] + space) * mid
            a = w / h**2 if h > 0 else 0
            for idx, layer in enumerate(layers):
                y = (size[1] + space) * (idx - mid)
                x = a * y**2
                layer.center = (x, y)
        else:
            w = (size[0] + space) * mid
            h = curve * orientation
            a = h / w**2 if w > 0 else 0
            for idx, layer in enumerate(layers):
                x = (size[0] + space) * (idx - mid)
                y = a * x**2
                layer.center = (x, y)

        if not upward:
            for idx, layer in enumerate(layers):
                quad = layer.quad
                x, y = layer.center
                if vertical:
                    angle = np.degrees(np.arctan(x / y)) if y != 0 else 0
                else:
                    angle = -np.degrees(np.arctan(y / x)) if x != 0 else 0
                matrix = cv2.getRotationMatrix2D((x, y), angle, 1)
                quad = np.append(quad, np.ones((4, 1)), axis=-1).dot(matrix.T)
                layer.quad = quad

        return meta
