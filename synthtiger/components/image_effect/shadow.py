"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np
import pytweening

from synthtiger import utils
from synthtiger.components.component import Component


class Shadow(Component):
    def __init__(
        self,
        intensity=(0, 192),
        amount=(0.5, 1),
        smoothing=(0, 1),
        bidirectional=0.5,
        align=(-1, 1),
        angle=(0, 360),
    ):
        super().__init__()
        self.intensity = intensity
        self.amount = amount
        self.smoothing = smoothing
        self.bidirectional = bidirectional
        self.align = align
        self.angle = angle

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        intensity = meta.get(
            "intensity", np.random.randint(self.intensity[0], self.intensity[1] + 1)
        )
        amount = meta.get("amount", np.random.uniform(self.amount[0], self.amount[1]))
        smoothing = meta.get(
            "smoothing", np.random.uniform(self.smoothing[0], self.smoothing[1])
        )
        bidirectional = meta.get("bidirectional", np.random.rand() < self.bidirectional)
        align = meta.get("align", np.random.uniform(self.align[0], self.align[1]))
        angle = meta.get("angle", np.random.uniform(self.angle[0], self.angle[1]))

        meta = {
            "intensity": intensity,
            "amount": amount,
            "smoothing": smoothing,
            "bidirectional": bidirectional,
            "align": align,
            "angle": angle,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        intensity = meta["intensity"]
        amount = meta["amount"]
        smoothing = meta["smoothing"]
        bidirectional = meta["bidirectional"]
        align = meta["align"]
        angle = meta["angle"]

        scale = abs(np.sin(np.deg2rad(angle))) + abs(np.cos(np.deg2rad(angle)))

        if bidirectional:
            start = (1 - amount) / 2 + (1 - amount) / 2 * align
            end = (1 + amount) / 2 + (1 - amount) / 2 * align
            grad_start = start - amount / 2 * smoothing
            grad_end = end + amount / 2 * smoothing
            peak_start = start + amount / 2 * smoothing
            peak_end = end - amount / 2 * smoothing
        else:
            start = 1 - amount
            end = 1
            grad_start = start - min(amount, 1 - amount) * smoothing
            grad_end = end
            peak_start = start + min(amount, 1 - amount) * smoothing
            peak_end = end

        for layer in layers:
            height, width = layer.image.shape[:2]
            size = max(width, height)
            values = np.zeros(size)

            start, end = int(size * grad_start), int(size * peak_start)
            left, right = np.clip(start, 0, size), np.clip(end, 0, size)
            grad_left, grad_right = max(left - start, 0), max(right - start, 0)
            grads = np.linspace(0, 1, end - start)[grad_left:grad_right]
            values[left:right] = [pytweening.easeInOutQuad(grad) for grad in grads]

            start, end = int(size * peak_start), int(size * peak_end)
            left, right = np.clip(start, 0, size), np.clip(end, 0, size)
            values[left:right] = 1

            start, end = int(size * peak_end), int(size * grad_end)
            left, right = np.clip(start, 0, size), np.clip(end, 0, size)
            grad_left, grad_right = max(left - start, 0), max(right - start, 0)
            grads = np.linspace(1, 0, end - start)[grad_left:grad_right]
            values[left:right] = [pytweening.easeInOutQuad(grad) for grad in grads]

            shadow = utils.create_image((size, size))
            shadow[..., 3] = values * intensity

            matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
            shadow = cv2.warpAffine(
                shadow,
                matrix,
                (size, size),
                borderMode=cv2.BORDER_REPLICATE,
            )
            shadow = cv2.resize(shadow, (width, height))

            image = utils.blend_image(shadow, layer.image)
            layer.image = image

        return meta
