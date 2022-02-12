"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from synthtiger.components.component import Component


class Iterator(Component):
    def __init__(self, components, args=None):
        super().__init__()
        self.components = components

        if args is not None:
            for component, arg in zip(self.components, args):
                component._init(**arg)

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        sub_metas = meta.get("metas", [{} for _ in self.components])

        meta = {
            "metas": [],
        }

        for idx, component in enumerate(self.components):
            meta["metas"].append(component.sample(sub_metas[idx]))

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        sub_metas = meta["metas"]

        for idx, component in enumerate(self.components):
            component.apply(layers, sub_metas[idx])

        return meta

    def data(self, meta):
        sub_metas = meta["metas"]
        data = []

        for idx, component in enumerate(self.components):
            data.append(component.data(sub_metas[idx]))

        return data

    def _init(self, *args, **kwargs):
        self.__init__(self.components, *args, **kwargs)
