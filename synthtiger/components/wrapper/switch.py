"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component


class Switch(Component):
    def __init__(self, component, prob=1, args=None):
        super().__init__()
        self.component = component
        self.prob = prob

        if args is not None:
            self.component._init(**args)

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        state = meta.get("state", np.random.rand() < self.prob)

        sub_meta = None
        if state:
            sub_meta = self.component.sample(meta.get("meta"))

        meta = {
            "state": state,
            "meta": sub_meta,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        state = meta["state"]
        sub_meta = meta["meta"]

        if state:
            self.component.apply(layers, sub_meta)

        return meta

    def data(self, meta):
        state = meta["state"]
        sub_meta = meta["meta"]

        data = None
        if state:
            data = self.component.data(sub_meta)

        return data

    def _init(self, *args, **kwargs):
        self.__init__(self.component, *args, **kwargs)
