"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger import utils
from synthtiger.components.component import Component


class Pad(Component):
    def __init__(self, pxs=None, percents=None, mode="constant", value=0):
        super().__init__()
        self.pxs = pxs
        self.percents = percents
        self.mode = mode
        self.value = value

        shapes = [(1, 2), (2, 2), (3, 2), (4, 2)]
        if self.pxs is not None and np.array(self.pxs).shape not in shapes:
            raise TypeError("Shape of pxs must be (1,2), (2,2), (3,2) or (4,2)")
        if self.percents is not None and np.array(self.percents).shape not in shapes:
            raise TypeError("Shape of percents must be (1,2), (2,2), (3,2) or (4,2)")

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
        mode = meta.get("mode", self.mode)
        value = meta.get("value", self.value)

        meta = {
            "pxs": pxs,
            "percents": percents,
            "mode": mode,
            "value": value,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        pxs = meta["pxs"]
        percents = meta["percents"]
        mode = meta["mode"]
        value = meta["value"]

        if pxs is not None:
            pxs = np.tile(pxs, 4)[:4]
        if percents is not None:
            percents = np.tile(percents, 4)[:4]

        for layer in layers:
            image = layer.output()
            height, width = image.shape[:2]

            margins = [0, 0, 0, 0]
            if pxs is not None:
                margins = pxs
            elif percents is not None:
                margins = percents * np.tile((height, width), 2)

            top, right, bottom, left = margins
            image = utils.pad_image(
                image,
                top=top,
                right=right,
                bottom=bottom,
                left=left,
                mode=mode,
                value=value,
            )

            topleft = layer.topleft - (left, top)
            height, width = image.shape[:2]

            layer.image = image
            layer.bbox = [*topleft, width, height]

        return meta
