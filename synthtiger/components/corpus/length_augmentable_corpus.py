"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.corpus.base_corpus import BaseCorpus


class LengthAugmentableCorpus(BaseCorpus):
    def __init__(
        self,
        paths=(),
        weights=(),
        min_length=None,
        max_length=None,
        charset=None,
        textcase=None,
        augmentation=0,
        augmentation_length=(1, 25),
    ):
        super().__init__(paths, weights, min_length, max_length, charset, textcase)
        self.augmentation = augmentation
        self.augmentation_length = augmentation_length

    def _sample_text(self):
        augmentation = np.random.rand() < self.augmentation
        if not augmentation:
            return super()._sample_text()

        text = ""
        length = np.random.randint(
            self.augmentation_length[0], self.augmentation_length[1] + 1
        )

        while len(text) < length:
            key = np.random.choice(len(self.paths), p=self._probs)
            if self._counts[key] == 0:
                raise RuntimeError(f"There is no text: {self.paths[key]}")

            idx = np.random.randint(self._counts[key])
            text += self._get_text(key, idx)

        text = text[:length]
        return text
