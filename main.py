#!/usr/bin/python

from os import stat
from typing import Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from numpy import ndarray, float64
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import kaiser
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
import warnings

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

    start = 0
    end = 6

    return graph, start, end


def getChildren(system: ndarray, index: int):
    for next in range(len(system)):
        edge = system[index, next]
        if edge != 0:
            yield next, edge


def getParents(system: ndarray, index: int):
    # parents:List[tuple(int, float64)] = []
    parents = []
    for parent in range(len(system)):
        edge = system[parent, index]
        if edge != 0:
            parents.append((parent, edge))
    return parents


def findPaths(system: ndarray, root: int):
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

    start = [root]
    return findPaths_r(system, start, [])


def simulatePaths(trials: int = 1, display: bool = False):
    # get number of paths
    system, start, end = getSystem()
    paths = findPaths(system, start)
    path_cnt = len(paths)

    history = np.empty((trials, path_cnt))

    np.random.seed(137)
    for trial in track(range(trials), 'Running Path Trials'):
        system, _, _ = getSystem()
        for path_id, path in enumerate(paths):
            time = 0
            for idx in range(len(path)-1):
                edge = path[idx], path[idx+1]
                time += system[edge]
            history[trial, path_id] = time

    # Create dataframe for visualization
    columns = [str(path) for path in paths]
    df = pd.DataFrame(history, columns=columns)

    #! Remove constant path because it distorts the bound
    df = df.drop(columns=columns[3])
    cols = columns.copy()
    cols.pop(3)

    if display:
        with console.status('Plotting/Displaying'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for col in track(cols, 'Plotting'):
                    sns.distplot(df[col])

            sns.set()
            plt.title('PDF for System Paths')
            plt.legend(labels=cols)
            plt.xlabel('Time')
            # plt.ylabel('')
            plt.tight_layout()  # type: ignore
            plt.show()

    return df[cols[-2]]


def simulateSystem(trials: int = 1, display: bool = False):
    def calcTime(system: ndarray, curr: int, root: int) -> float64:
        parents = getParents(system, curr)

        def add_parents_and_edge(idx) -> float64:
            time = calcTime(system, idx, root)
            time += system[idx, curr]
            return time

        # if the start is the only parent return the edge value
        if root == parents[0][0] and len(parents) == 1:
            return parents[0][1]
        # else continue to bubble up
        else:
            times = [add_parents_and_edge(p[0]) for p in parents]
            return max(times)

    history = np.empty(trials)

    for trial in track(range(trials), 'Running System Trials'):
        system, start, end = getSystem()
        history[trial] = calcTime(system, end, start)

    df = pd.DataFrame(history, columns=['Time'])

    if display:
        with console.status('Plotting/Displaying'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sns.distplot(df)
            sns.set()
            plt.title('PDF for System Simulation')
            plt.xlabel('Time')
            # plt.ylabel('')
            plt.tight_layout()  # type: ignore
            plt.show()

    return df


def main(args: Namespace):
    trials = args.trials
    # trials = 1000000
    trials = 10000

    display = True
    critical_path = simulatePaths(trials, display)
    sim_time = simulateSystem(trials, display)

    with console.status('Plotting/Displaying'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.distplot(critical_path, label='Critical Path')
            sns.distplot(sim_time, label='System')
        sns.set()
        plt.title('Simulation PDFs')
        plt.xlabel('Time')
        # plt.ylabel('')
        plt.legend()
        plt.tight_layout()  # type: ignore
        plt.show()

    pass


if __name__ == '__main__':
    args = setupArgParser()
    main(args)
