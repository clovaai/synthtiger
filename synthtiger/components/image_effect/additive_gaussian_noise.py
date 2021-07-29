"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import imgaug.augmenters as iaa
import numpy as np

from synthtiger.components.component import Component


class AdditiveGaussianNoise(Component):
    def __init__(self, scale=(8, 32), per_channel=0.5):
        super().__init__()
        self.scale = scale
        self.per_channel = per_channel

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        scale = meta.get("scale", np.random.uniform(self.scale[0], self.scale[1]))
        per_channel = meta.get("per_channel", np.random.rand() < self.per_channel)

        meta = {
            "scale": scale,
            "per_channel": per_channel,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        scale = meta["scale"]
        per_channel = meta["per_channel"]
        aug = iaa.AdditiveGaussianNoise(scale=scale, per_channel=per_channel)

        for layer in layers:
            rgb = layer.image[..., :3].astype(np.uint8)
            alpha = layer.image[..., 3, np.newaxis].astype(np.uint8)
            rgb = aug(image=rgb)
            image = np.concatenate((rgb, alpha), axis=-1).astype(np.float32)
            layer.image = image

        return meta
