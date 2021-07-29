"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from synthtiger.components.corpus.base_corpus import BaseCorpus
from synthtiger.components.corpus.char_augmentable_corpus import CharAugmentableCorpus
from synthtiger.components.corpus.length_augmentable_corpus import (
    LengthAugmentableCorpus,
)

__all__ = ["BaseCorpus", "CharAugmentableCorpus", "LengthAugmentableCorpus"]
