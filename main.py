#!/usr/bin/python

from os import stat
from typing import Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from numpy import ndarray, float64
from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser, Namespace
import cupy as cp

from rich import inspect, print as p
from rich.console import Console
from rich.syntax import Syntax, SyntaxTheme
from rich import traceback
from rich.progress import track

sns.set()
console = Console()
traceback.install()

NP_TYPE = cp.byte

PROMPT = """PROMPT
A software process is represented as a network with some process time between nodes having Uniform distribution (U) and others with Deterministic values.
  c.Analyze the performance of the system,
  d.After adequate samples, show how you quantify the criticalityof each path.
  e.Briefly explain your redesign perspectiveof such a system
"""

WEIGHTS = """
(1, 2):  U (4,6)
(1, 5):  6   
(2, 3):  6
(2, 4):  U (6,8)
(3, 4):  Triangle(4,8,10)
(4, 7):  4
(5, 3):  8   
(5, 4):  11
(5, 6):  U (8,10)
(6, 7):  U (9, 10)
"""


def title(str: Any):
    console.rule(str)


def log(str: Any):
    console.log(str)


def toPercent(num) -> str:
    return f'{np.round(num * 100, 2)}%'


def setupArgParser() -> Namespace:
    parser = ArgumentParser(description=PROMPT)
    # parser.add_argument('-trials', metavar='T', type=int, default=120000000,  # Max size for random
    parser.add_argument('-trials', metavar='T', type=int, default=1,  # Max size for random
                        help='Number of trials to run (to stabilize statistical values)')
    return parser.parse_args()


def getUniform(min: float, max: float):
    return np.random.uniform(min, max)   # type: ignore


def getTriangle(left, mode, right):
    return np.random.triangular(left, mode, right)   # type: ignore


def getSystem():
    # np.random.seed(137)
    graph = np.zeros((7, 7))
    graph[0, 1] = getUniform(4, 6)
    graph[0, 4] = 6
    graph[1, 2] = 6
    graph[1, 3] = getUniform(6, 8)
    graph[2, 3] = getTriangle(4, 8, 10)
    graph[3, 6] = 4
    graph[4, 2] = 8
    graph[4, 3] = 11
    graph[4, 5] = getUniform(8, 10)
    graph[5, 6] = getUniform(9, 10)
    starts = [0]
    ends = [6]
    return graph, starts, ends


def findPaths(system: ndarray, root: int):
    start = [root]
    return findPaths_r(system, start, [])


def findPaths_r(system: ndarray, path: List[int], paths: List[List[int]]):
    root = path[-1]
    # p(f'[{root+ 1}]')

    if root == len(system) - 1:
        paths += [path]
        # log(paths)
    else:
        for next in range(len(system)):
            edge = system[root, next]
            if edge != 0:
                # p(f'  ({root + 1}, {next + 1}) -> {edge.round(2)}')
                paths = findPaths_r(system, path + [next], paths)

    return paths


def main(args: Namespace):
    trials = args.trials
    trials = 100000

    # get number of paths
    system, starts, _ = getSystem()
    paths = findPaths(system, starts[0])
    path_cnt = len(paths)

    history = np.zeros((trials, path_cnt))

    np.random.seed(137)

    for trial in track(range(trials)):
        # system, starts, ends = getSystem()
        system, _, _ = getSystem()
        # paths = findPaths(system, starts[0])
        for path_id, path in enumerate(paths):
            time = 0
            for idx in range(len(path)-1):
                edge = path[idx], path[idx+1]
                time += system[edge]
            # p(f'path {path}:\n    time = {time.round(2)}')
            history[trial, path_id] = time

    # Create dataframe for visualization
    columns = [str(path) for path in paths]
    df = pd.DataFrame(history, columns=columns)
    # df /= df.max(axis=1)
    p(df.head())

    #! Remove constant path because it distorts the bound
    df = df.drop(columns=columns[3])

    # sns.histplot(df, bins=100, stat='probability')
    sns.histplot(df, bins=50)
    # sns.kdeplot(data=df)
    # plt.show()
    # sns.histplot(df[columns[0]], bins=100, stat='probability')
    # plt.show()
    # sns.histplot(df[columns[1]], bins=100, stat='probability')
    # plt.show()
    # sns.histplot(df[columns[2]], bins=100, stat='probability')
    # plt.show()
    # sns.histplot(df[columns[4]], bins=100, stat='probability')
    # plt.show()
    # df = pd.Series(history[:, [0, 1, 2, 4]].flatten())   # type: ignore
    # sns.histplot(df, bins=50, stat='probability')
    # sns.kdeplot(data=df)
    # sns.distplot(df)
    plt.show()
    pass


if __name__ == '__main__':
    args = setupArgParser()
    main(args)
