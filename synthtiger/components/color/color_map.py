"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component


class ColorMap(Component):
    def __init__(self, paths=(), weights=(), k=2):
        super().__init__()
        self.paths = paths
        self.weights = weights
        self.k = k
        self._cluster_groups = []
        self._counts = []
        self._probs = np.array(self.weights) / sum(self.weights)
        self._update_cluster_groups()

    def _update_cluster_groups(self):
        self._cluster_groups = []
        self._counts = []

        for path in self.paths:
            cluster_group = []

            with open(path, "r", encoding="utf-8") as fp:
                for row in fp:
                    values = row.split()

                    k = len(values) // 2
                    if k != self.k:
                        continue

                    clusters = []
                    for idx in range(0, len(values), 2):
                        center, std = values[idx], values[idx + 1]
                        center = list(map(float, center.split(",")))
                        std = float(std)
                        clusters.append((center, std))

                    cluster_group.append(clusters)

            self._cluster_groups.append(cluster_group)
            self._counts.append(len(cluster_group))

    def _sample_colormap(self):
        key = np.random.choice(len(self.paths), p=self._probs)
        if self._counts[key] == 0:
            raise RuntimeError(f"There is no colormap: {self.paths[key]}")

        cluster_group = self._cluster_groups[key]
        clusters = cluster_group[np.random.randint(len(cluster_group))]
        colormap = [np.random.normal(center, std) for center, std in clusters]
        colormap = np.random.permutation(colormap)
        return colormap
