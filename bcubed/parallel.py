# Simple extended BCubed implementation in Python for clustering evaluation
# Copyright 2020 Hugo Hromic, Chris Bowdon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Extended BCubed algorithm taken from:
# Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation metrics
# based on formal constraints." Information retrieval 12.4 (2009): 461-486.

"""Generate extended BCubed evaluation for clustering."""

"""Parallelized versions of functions in bcubed.extended."""
import numpy
from multiprocessing import Pool, cpu_count
from itertools import repeat
from .extended import mult_precision, mult_recall

def _p(el1, cdict, ldict):
    return numpy.mean([mult_precision(el1, el2, cdict, ldict)
                       for el2 in cdict if cdict[el1] & cdict[el2]])

def _r(el1, cdict, ldict):
    return numpy.mean([mult_recall(el1, el2, cdict, ldict)
                       for el2 in cdict if ldict[el1] & ldict[el2]])

def parallel(function, cdict, ldict, n_processes=None):
    if n_processes is None:
        n_processes = max(1, cpu_count() - 2)

    with Pool(n_processes) as pool:
        return pool.starmap(function, zip(cdict.keys(), repeat(cdict), repeat(ldict)))

def precision(cdict, ldict, n_processes=None):
    """Computes overall extended BCubed precision for the C and L dicts
    using multiple processes for parallelism.

    Parameters
    ==========
    cdict: dict(item: set(cluster-ids))
        The cluster assignments to be evaluated
    ldict: dict(item: set(cluster-ids))
        The ground truth clustering
    n_processes: optional integer
        Number of processes to use (defaults to number of CPU cores - 1)
    """
    p_per_el = parallel(_p, cdict, ldict, n_processes)
    return numpy.mean(p_per_el)

def recall(cdict, ldict, n_processes=None):
    """Computes overall extended BCubed recall for the C and L dicts
    using multiple processes for parallelism.

    Parameters
    ==========
    cdict: dict(item: set(cluster-ids))
        The cluster assignments to be evaluated
    ldict: dict(item: set(cluster-ids))
        The ground truth clustering
    n_processes: optional integer
        Number of processes to use (defaults to number of CPU cores - 1)
    """
    r_per_el = parallel(_r, cdict, ldict, n_processes)
    return numpy.mean(r_per_el)
