"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np

from synthtiger.components.component import Component


class TextSprinkle(Component):
    def __init__(self, prob=(0.05, 0.2), offset=(-2, 2), masking=0.5):
        super().__init__()
        self.prob = prob
        self.offset = offset
        self.masking = masking

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        prob = meta.get("prob", np.random.uniform(self.prob[0], self.prob[1]))
        offset = meta.get("offset", (self.offset[0], self.offset[1]))
        masking = meta.get("masking", np.random.rand() < self.masking)

        meta = {
            "prob": prob,
            "offset": offset,
            "masking": masking,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        prob = meta["prob"]
        offset = meta["offset"]
        masking = meta["masking"]

        for layer in layers:
            height, width = layer.image.shape[:2]

            count = int(height * width * prob)
            idxes = np.random.randint(height * width, size=count)
            offsets = np.random.uniform(offset[0], offset[1], (count, 2))

            dx = np.zeros(height * width, dtype=np.float32)
            dy = np.zeros(height * width, dtype=np.float32)
            dx[idxes] = offsets[..., 0]
            dy[idxes] = offsets[..., 1]
            dx = np.reshape(dx, (height, width))
            dy = np.reshape(dy, (height, width))

            alpha = layer.image[..., 3] / 255 if masking else 1
            mapy, mapx = np.indices((height, width), dtype=np.float32)
            mapx += dx * alpha
            mapy += dy * alpha
            out_image = cv2.remap(layer.image, mapx, mapy, cv2.INTER_LINEAR)

            layer.image = out_image

        return meta
