"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import imgaug.augmenters as iaa
import numpy as np

from synthtiger.components.component import Component


class JpegCompression(Component):
    def __init__(self, compression=(5, 30)):
        super().__init__()
        self.compression = compression

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        compression = meta.get(
            "compression",
            np.random.randint(self.compression[0], self.compression[1] + 1),
        )

        meta = {
            "compression": compression,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        compression = meta["compression"]
        aug = iaa.JpegCompression(compression=compression)

        for layer in layers:
            rgb = layer.image[..., :3].astype(np.uint8)
            alpha = layer.image[..., 3, np.newaxis].astype(np.uint8)
            rgb = aug(image=rgb)
            image = np.concatenate((rgb, alpha), axis=-1).astype(np.float32)
            layer.image = image

        return meta
