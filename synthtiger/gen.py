"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
import random
import sys
import traceback
from multiprocessing import Process, Queue

import numpy as np
import yaml


def read_template(path, name, config=None):
    path = os.path.abspath(path)
    root = os.path.dirname(path)
    module = os.path.splitext(os.path.basename(path))[0]
    sys.path.append(root)
    template = getattr(__import__(module), name)(config)
    sys.path.remove(root)
    del sys.modules[module]
    return template


def read_config(path):
    with open(path, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    return config


def generator(path, name, config=None, worker=0, verbose=False):
    if worker > 0:
        queue = Queue(maxsize=1024)
        for _ in range(worker):
            _run(_worker, (path, name, config, queue, verbose))

        while True:
            data = queue.get()
            yield data
    else:
        template = read_template(path, name, config)

        while True:
            data = _generate(template, verbose)
            yield data


def _run(func, args):
    proc = Process(target=func, args=args)
    proc.daemon = True
    proc.start()
    return proc


def _worker(path, name, config, queue, verbose):
    random.seed()
    np.random.seed()
    template = read_template(path, name, config)

    while True:
        data = _generate(template, verbose)
        queue.put(data)


def _generate(template, verbose):
    while True:
        try:
            data = template.generate()
        except:
            if verbose:
                print(f"{traceback.format_exc()}")
            continue
        return data
