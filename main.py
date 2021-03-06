#!/usr/bin/python

import warnings
from argparse import ArgumentParser, Namespace
from types import GeneratorType
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import float64, ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from rich import inspect
from rich import print as p
from rich import traceback
from rich.console import Console
from rich.progress import track

# Global console for pretty printing and traceback for better debugging
console = Console()
traceback.install()

PROMPT = """PROMPT
A software process is represented as a network with some process time between \
nodes having Uniform distribution (U) and others with Deterministic values.
  [c. Analyze the performance of the system],
  [d. After adequate samples, show how you quantify the criticalityof each path],
  [e. Briefly explain your redesign perspectiveof such a system]
"""


def setupArgParser() -> Namespace:
    parser = ArgumentParser(description=PROMPT)
    parser.add_argument('-trials', metavar='T', type=int, default=10,
                        help='Number of trials to run (to stabilize statistical values)')
    parser.add_argument('-use-kitty', action='store_true', help='Allows inline plotting in the Kitty terminal')
    parser.add_argument('-plots', action='store_true')
    return parser.parse_args()


def getSystem() -> Tuple[ndarray, int, int]:
    # np.random.seed(137)

    Uniform = np.random.uniform
    Triangle = np.random.triangular  # type: ignore

    graph = np.zeros((7, 7))
    graph[0, 1] = Uniform(4, 6)  # type: ignore
    graph[0, 4] = 6
    graph[1, 2] = 6
    graph[1, 3] = Uniform(6, 8)  # type: ignore
    graph[2, 3] = Triangle(4, 8, 10)
    graph[3, 6] = 4
    graph[4, 2] = 8
    graph[4, 3] = 11
    graph[4, 5] = Uniform(8, 10)  # type: ignore
    graph[5, 6] = Uniform(9, 10)  # type: ignore

    start = 0
    end = 6

    return graph, start, end


def getChildren(system: ndarray, index: int) -> GeneratorType:
    for next in range(len(system)):
        edge = system[index, next]
        if edge != 0:
            yield next


def getParents(system: ndarray, index: int) -> List[Tuple[int, float64]]:
    parents: List[Tuple[int, float64]] = []
    for parent in range(len(system)):
        edge = system[parent, index]
        if edge != 0:
            parents.append((parent, edge))
    return parents


def findPaths(system: ndarray, root: int) -> List[List[int]]:
    def findPaths_r(system: ndarray, path: List[int], paths: List[List[int]]):
        root = path[-1]

        if root == len(system) - 1:
            paths += [path]
        else:
            for child in getChildren(system, root):
                paths = findPaths_r(system, path + [child], paths)

        return paths

    start = [root]
    return findPaths_r(system, start, [])


def simulatePaths(trials: int = 1, display: bool = False) -> Series:
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
    df = DataFrame(history, columns=columns)

    # Print results
    for col in columns:
        p(f'Path {col}\t-> mean: {df[col].mean()}')
    p(f'Critical Path = {columns[2]}')

    #! Remove constant path because it distorts the bound
    df = df.drop(columns=columns[3])
    cols = columns.copy()
    cols.pop(3)

    if display:
        with console.status('Plotting'), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for col in track(cols, 'Plotting'):
                sns.distplot(df[col])
            sns.set()
            plt.title('PDF for System Paths')
            plt.legend(labels=cols)
            plt.xlabel('Time')
            plt.tight_layout()  # type: ignore
        plt.show()

    return df[columns[2]]


def simulateSystem(trials: int = 1, display: bool = False) -> Series:
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

    np.random.seed(137)
    for trial in track(range(trials), 'Running System Trials'):
        system, start, end = getSystem()
        history[trial] = calcTime(system, end, start)

    hist = Series(history)

    # Print results
    p('[#10FF99][Trials stats]:')
    p(hist.describe())   # type: ignore

    if display:
        with console.status('Plotting'), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.distplot(hist)
            sns.set()
            plt.title('PDF for System Simulation')
            plt.xlabel('Time')
            plt.tight_layout()  # type: ignore
        plt.show()

    return hist


def main(args: Namespace):
    trials = args.trials

    display = args.plots
    console.rule('[bold purple]Paths Simulation')
    critical_path = simulatePaths(trials, display)
    console.rule('[bold blue]System Simulation')
    sim_time = simulateSystem(trials, display)

    console.rule('[bold cyan]Critical Path vs. System Simulation')

    if display:
        with console.status('Plotting'), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.distplot(critical_path, label='Critical Path')
            sns.distplot(sim_time, label='System')
            sns.set()
            plt.title('Simulation PDFs')
            plt.xlabel('Time')
            plt.legend()
            plt.tight_layout()  # type: ignore
        plt.show()

    expected_val = critical_path.mean()
    actual_val = sim_time.mean()
    pct_error = abs((actual_val - expected_val)/expected_val) * 100
    p(f'Critical Path Mean:    {expected_val:0.4f}')
    p(f'Simulated System Mean: {actual_val  :0.4f}')
    p(f'Error: {pct_error:0.2f}%')


if __name__ == '__main__':
    args = setupArgParser()

    if args.use_kitty:
        mpl.use('module://matplotlib-backend-kitty')
        mpl.rc('figure', figsize=(100, 1), dpi=100)   # type: ignore

    main(args)
