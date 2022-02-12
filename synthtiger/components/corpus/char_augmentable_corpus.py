"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from collections import Counter

import numpy as np

from synthtiger import utils
from synthtiger.components.corpus.base_corpus import BaseCorpus


class CharAugmentableCorpus(BaseCorpus):
    def __init__(
        self,
        paths=(),
        weights=(),
        min_length=None,
        max_length=None,
        charset=None,
        textcase=None,
        augmentation=0,
        augmentation_charset=None,
    ):
        super().__init__(paths, weights, min_length, max_length, charset, textcase)
        self.augmentation = augmentation
        self.augmentation_charset = augmentation_charset
        self._augmentation_charset = set()
        self._dists = []
        self._update_dists()

    def _update_dists(self):
        self._charset = set()
        self._dists = []

        if self.augmentation > 0:
            self._augmentation_charset = utils.read_charset(self.augmentation_charset)

        for key in range(len(self.paths)):
            count = self._counts[key]
            if self.augmentation == 0:
                dist = np.ones(count, dtype=np.float32)
            else:
                char_count = Counter()
                for idx in range(count):
                    text = self._get_text(key, idx)
                    char_count.update(set(text))

                dist = np.empty(count, dtype=np.float32)
                for idx in range(count):
                    text = self._get_text(key, idx)
                    charset = set(text) & self._augmentation_charset
                    dist[idx] = sum([1 / char_count[char] for char in charset])

            total = np.sum(dist)
            if total > 0:
                dist /= total
            dist = np.cumsum(dist)
            self._dists.append(dist)

    def _sample_text(self):
        augmentation = np.random.rand() < self.augmentation
        if not augmentation:
            return super()._sample_text()

        key = np.random.choice(len(self.paths), p=self._probs)
        if self._counts[key] == 0:
            raise RuntimeError(f"There is no text: {self.paths[key]}")

        value = np.random.rand() * self._dists[key][-1]
        idx = np.searchsorted(self._dists[key], value)
        idx = min(idx, self._counts[key] - 1)
        text = self._get_text(key, idx)
        return text
