"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import pytest

import synthtiger


@pytest.mark.parametrize(
    "args",
    [
        "synthtiger_horizontal_args",
        "synthtiger_vertical_args",
        "multiline_args",
    ],
)
def test_generation_error(args, request):
    """Test for errors during data generation"""

    args = request.getfixturevalue(args)
    seed = None

    synthtiger.set_global_random_seed(seed)
    config = synthtiger.read_config(args["config"])
    generator = synthtiger.generator(
        args["script"],
        args["name"],
        config=config,
        count=None,
        worker=0,
        seed=seed,
        retry=False,
        verbose=True,
    )

    total = 100
    error = 0

    for _ in range(total):
        _, data = next(generator)
        if data is None:
            error += 1

    assert error < total
