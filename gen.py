"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import argparse
import pprint
import time

import synthtiger


def run(args):
    if args.config is not None:
        config = synthtiger.read_config(args.config)
    pprint.pprint(config)

    template = synthtiger.read_template(args.script, args.name, config)
    generator = synthtiger.generator(
        args.script, args.name, config, worker=args.worker, verbose=args.verbose
    )

    if args.output is not None:
        template.init_save(args.output)

    for idx in range(args.count):
        data = next(generator)
        if args.output is not None:
            template.save(args.output, data, idx)
        print(f"Generated {idx + 1} data")

    if args.output is not None:
        template.end_save(args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        type=str,
        help="Directory path to save data.",
    )
    parser.add_argument(
        "-c",
        "--count",
        metavar="INTEGER",
        type=int,
        default=100,
        help="Number of data. [default: 100]",
    )
    parser.add_argument(
        "-w",
        "--worker",
        metavar="INTEGER",
        type=int,
        default=0,
        help="Number of workers. If 0, It generates data in the main process. [default: 0]",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print error messages while generating data.",
    )
    parser.add_argument(
        "script",
        metavar="SCRIPT",
        type=str,
        help="Script file path.",
    )
    parser.add_argument(
        "name",
        metavar="NAME",
        type=str,
        help="Template class name.",
    )
    parser.add_argument(
        "config",
        metavar="CONFIG",
        type=str,
        nargs="?",
        help="Config file path.",
    )
    args = parser.parse_args()

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
