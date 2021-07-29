"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import imgaug.augmenters as iaa
import numpy as np

from synthtiger.components.component import Component


class CoarseDropout(Component):
    def __init__(
        self,
        p=(0.05, 0.25),
        size_px=None,
        size_percent=None,
        per_channel=0.5,
        only_alpha=0,
    ):
        super().__init__()
        self.p = p
        self.size_px = size_px
        self.size_percent = size_percent
        self.per_channel = per_channel
        self.only_alpha = only_alpha

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        p = meta.get("p", np.random.uniform(self.p[0], self.p[1]))
        per_channel = meta.get("per_channel", np.random.rand() < self.per_channel)
        only_alpha = meta.get("only_alpha", np.random.rand() < self.only_alpha)

        size_px = None
        if self.size_px is not None:
            size_px = np.random.randint(self.size_px[0], self.size_px[1] + 1)
        size_px = meta.get("size_px", size_px)

        size_percent = None
        if self.size_percent is not None:
            size_percent = np.random.uniform(self.size_percent[0], self.size_percent[1])
        size_percent = meta.get("size_percent", size_percent)

        meta = {
            "p": p,
            "size_px": size_px,
            "size_percent": size_percent,
            "per_channel": per_channel,
            "only_alpha": only_alpha,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        p = meta["p"]
        size_px = meta["size_px"]
        size_percent = meta["size_percent"]
        per_channel = meta["per_channel"]
        only_alpha = meta["only_alpha"]
        aug = iaa.CoarseDropout(
            p=p, size_px=size_px, size_percent=size_percent, per_channel=per_channel
        )

        for layer in layers:
            rgb = layer.image[..., :3].astype(np.uint8)
            alpha = layer.image[..., 3, np.newaxis].astype(np.uint8)
            if not only_alpha:
                rgb = aug(image=rgb)
            else:
                alpha = aug(image=alpha)
            image = np.concatenate((rgb, alpha), axis=-1).astype(np.float32)
            layer.image = image

        return meta
