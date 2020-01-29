import bcubed
import bcubed.parallel


def test_precision():
    cdict = {0: {0}, 1: {1}, 2: {0}, 3: {1}}
    ldict = {0: {1}, 1: {0}, 2: {1}, 3: {0}}
    assert bcubed.precision(cdict, ldict) == 1

    cdict = {0: {0}, 1: {1}, 2: {0}, 3: {1}}
    ldict = {0: {0}, 1: {1}, 2: {2}, 3: {1}}
    assert bcubed.precision(cdict, ldict) == 0.75


def test_recall():
    cdict = {0: {0}, 1: {1}, 2: {0}, 3: {1}}
    ldict = {0: {0}, 1: {1}, 2: {1}, 3: {1}}
    assert bcubed.recall(cdict, ldict) == 2/3

    cdict = {0: {0}, 1: {1}, 2: {0}, 3: {1}}
    ldict = {0: {0}, 1: {1}, 2: {2}, 3: {1}}
    assert bcubed.recall(cdict, ldict) == 1


def test_parallel_precision():
    cdict = {0: {0}, 1: {1}, 2: {0}, 3: {1}}
    ldict = {0: {1}, 1: {0}, 2: {1}, 3: {0}}
    assert bcubed.parallel.precision(cdict, ldict) == 1

    cdict = {0: {0}, 1: {1}, 2: {0}, 3: {1}}
    ldict = {0: {0}, 1: {1}, 2: {2}, 3: {1}}
    assert bcubed.parallel.precision(cdict, ldict) == 0.75

    cdict = {i: {i} for i in range(5000)}  # just enough to make it worth it
    ldict = {i: {i} for i in range(5000)}
    assert bcubed.parallel.precision(cdict, ldict) == 1


def test_parallel_recall():
    cdict = {0: {0}, 1: {1}, 2: {0}, 3: {1}}
    ldict = {0: {0}, 1: {1}, 2: {1}, 3: {1}}
    assert bcubed.parallel.recall(cdict, ldict) == 2/3

    cdict = {0: {0}, 1: {1}, 2: {0}, 3: {1}}
    ldict = {0: {0}, 1: {1}, 2: {2}, 3: {1}}
    assert bcubed.parallel.recall(cdict, ldict) == 1

    cdict = {i: {i} for i in range(5000)}  # just enough to make it worth it
    ldict = {i: {i} for i in range(5000)}
    assert bcubed.parallel.recall(cdict, ldict) == 1
