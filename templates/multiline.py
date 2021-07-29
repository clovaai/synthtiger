"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from synthtiger import components, layers


class Template:
    def __init__(self, **config):
        self.corpus = components.BaseCorpus(**config.get("corpus", {}))
        self.font = components.BaseFont(**config.get("font", {}))

    def generate(self):
        top_label = self.corpus.data(self.corpus.sample())
        bottom_label = self.corpus.data(self.corpus.sample())
        label = "\t".join([top_label, bottom_label])

        font = self.font.sample({"text": top_label + bottom_label})

        top_layer = layers.TextLayer(top_label, **font)
        bottom_layer = layers.TextLayer(bottom_label, **font)
        bottom_layer.center = top_layer.center
        bottom_layer.top = top_layer.bottom

        image = layers.Group([top_layer, bottom_layer]).output()

        data = {
            "image": image,
            "label": label,
            "ext": "png",
        }

        return data
