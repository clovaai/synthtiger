"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component


class Selector(Component):
    def __init__(self, components, weights=None, args=None):
        super().__init__()
        self.components = components
        self.weights = weights
        if self.weights is None:
            self.weights = [1] * len(components)
        self._probs = np.array(self.weights) / sum(self.weights)

        if args is not None:
            for component, arg in zip(self.components, args):
                component._init(**arg)

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        idx = meta.get("idx", self._sample_idx())
        sub_meta = self.components[idx].sample(meta.get("meta"))

        meta = {
            "idx": idx,
            "meta": sub_meta,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        idx = meta["idx"]
        sub_meta = meta["meta"]

        self.components[idx].apply(layers, sub_meta)

        return meta

    def data(self, meta):
        idx = meta["idx"]
        sub_meta = meta["meta"]
        data = self.components[idx].data(sub_meta)
        return data

    def _init(self, *args, **kwargs):
        self.__init__(self.components, *args, **kwargs)

    def _sample_idx(self):
        idx = np.random.choice(len(self.components), replace=False, p=self._probs)
        return idx
