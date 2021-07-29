"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import argparse
import os
import pprint
import random
import time
import traceback
from multiprocessing import Queue

import numpy as np

import utils


def produce(args, req):
    for _ in range(args.count):
        req.put(None)


def consume(args, req, res):
    random.seed()
    np.random.seed()

    template = utils.read_template(args.template)
    template = template(**args.config)

    while True:
        req.get()

        while True:
            try:
                data = template.generate()
            except:
                print(f"{traceback.format_exc()}")
                continue

            res.put(data)
            break


def run(args):
    req = Queue(maxsize=1024)
    res = Queue(maxsize=1024)

    producer = utils.run_process(produce, (args, req))
    consumers = []
    for _ in range(args.worker):
        consumer = utils.run_process(consume, (args, req, res))
        consumers.append(consumer)

    gt_path = os.path.join(args.output, "gt.txt")
    gt_file = utils.create_gt(gt_path)

    for idx in range(args.count):
        data = res.get()
        image = data["image"]
        label = data["label"]
        ext = data.get("ext", "png")
        quality = data.get("quality", 95)

        shard = str(idx // 10000)
        image_key = os.path.join("images", shard, f"{idx}.{ext}")
        image_path = os.path.join(args.output, image_key)

        utils.write_image(image_path, image, quality=quality)
        utils.write_gt(gt_file, image_key, label)
        print(f"Saved {idx + 1} images")

    gt_file.close()

    producer.terminate()
    for consumer in consumers:
        consumer.terminate()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--worker", type=int, default=1)
    args = parser.parse_args()

    args.config = utils.read_config(args.config)
    pprint.pprint(vars(args))

    return args


def main():
    start_time = time.time()
    args = parse_args()
    run(args)
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")


if __name__ == "__main__":
    main()
