"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component


class ColorMap(Component):
    def __init__(self, paths=None, weights=None):
        super().__init__()
        self.paths = [] if paths is None else paths
        self.weights = [1] * len(self.paths) if weights is None else weights
        self._probs = np.array(self.weights) / sum(self.weights)
        self._cluster_groups = []
        self._update_cluster_groups()

    def _update_cluster_groups(self):
        self._cluster_groups = []

        for path in self.paths:
            cluster_group = {}

            with open(path, "r", encoding="utf-8") as fp:
                for row in fp:
                    values = row.split()

                    clusters = []
                    for idx in range(0, len(values), 2):
                        center, std = values[idx], values[idx + 1]
                        center = list(map(float, center.split(",")))
                        std = float(std)
                        clusters.append((center, std))

                    k = len(values) // 2
                    if k not in cluster_group:
                        cluster_group[k] = []
                    cluster_group[k].append(clusters)

            self._cluster_groups.append(cluster_group)

    def _sample_colormap(self, key, k):
        cluster_group = self._cluster_groups[key].get(k)
        clusters = cluster_group[np.random.randint(len(cluster_group))]
        colormap = [np.random.normal(center, std) for center, std in clusters]
        colormap = np.random.permutation(colormap)
        return colormap
