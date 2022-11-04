import sys
from collections import OrderedDict, namedtuple
from datetime import datetime
from itertools import product
from os import path

import numpy as np
import pandas as pd
from sympy import divisors

from baseline_descriptor import BaselineDescriptor

from basic_lbp_descriptor import BasicLBP
from multiscale_lbp_descriptor import MultiscaleLBP
from random_descriptor import RandomDescriptor
from scikit_lbp_descriptor import ScikitLBP
from evaluation import evaluate_descriptor
from load_data import load_data


class RunBuilder:
    @staticmethod
    def get_runs(parameters: OrderedDict):
        """
        Function returns all permutations of parameters.
        :param parameters: Dictionary of parameters
        :return: List of all parameter permutations.
        """

        Run = namedtuple("Run", parameters.keys())
        runs = []

        for val in product(*parameters.values()):
            runs.append(Run(*val))

        return runs


class TaskRunner:
    @staticmethod
    def run(task_number: int):
        """
        Function runs the specified task

        :param task_number: Number of task to run
        """

        # report: https://www.overleaf.com/project/6346c7a6db473453dc5430d9
        data_dir = "./data/awe/"
        results_dir = "./results/"

        # Load data
        IMG_SIZE = 128
        print("Loading data...")
        X, y = load_data(data_dir, IMG_SIZE, IMG_SIZE, normalize=False)
        print(f"X.shape: {X.shape}")
        print(f"y.shape: {y.shape}")

        tile_sizes = list(divisors(X[0].shape[0], generator=True))[1: -1]

        if task_number == 0:
            # Evaluate random descriptor
            # Evaluate baseline descriptor
            accs = []
            for i in range(10):
                acc = evaluate_descriptor(RandomDescriptor,
                                          "manhattan",
                                          X, y,
                                          tile_sizes[3],
                                          verbose=False)
                accs.append(acc)

            print(f"Rank1 (Random): {np.mean(accs)}")

        if task_number == 1:
            # Evaluate baseline descriptor
            evaluate_descriptor(BaselineDescriptor,
                                "manhattan",
                                X, y,
                                tile_sizes[3])

        elif task_number == 2:
            # Evaluate basic LBP (P=8, R=1) descriptor.
            evaluate_descriptor(BasicLBP(to_hist=False),
                                "euclidean",
                                X, y,
                                tile_sizes[3])

        elif task_number == 3:
            # Evaluate Scikit implementation of LBP descriptor.
            evaluate_descriptor(ScikitLBP(num_points=8, radius=1, to_hist=True, method="uniform"),
                                "euclidean",
                                X, y,
                                tile_sizes[3])
        elif task_number == 4:
            # Evaluate Multiscale LBP descriptor implementation.
            evaluate_descriptor(MultiscaleLBP(num_points=8, radius=1, to_hist=True, method="uniform"),
                                "euclidean",
                                X, y,
                                tile_sizes[5])

        elif task_number == 5:
            # Evaluate Multiscale LBP  & ScikitLBP using a predefined set of hyper-parameters.
            hyper_parameters = OrderedDict(
                scale=[(4, 1), (8, 1), (12, 2), (16, 2)],
                distance_metric=["euclidean", "cosine", "manhattan"],
                to_hist=[True],
                tile_size=[16, 32]  # tile sizes for 128 x 128 image size: [2, 4, 8, 16, 32, 64]
            )
            run_data = []
            run_params = RunBuilder.get_runs(hyper_parameters)
            run_time = datetime.now()

            # Try each combination of parameters
            print(f"Running {len(run_params)} experiments...")
            for p_set in run_params:
                # Construct and evaluate descriptor.

                descriptor = MultiscaleLBP(num_points=p_set.scale[0], radius=p_set.scale[1],
                                           to_hist=p_set.to_hist, method="uniform")
                #descriptor = ScikitLBP(num_points=p_set.scale[0], radius=p_set.scale[1],
                #                       to_hist=p_set.to_hist, method="default")

                acc = evaluate_descriptor(descriptor, p_set.distance_metric, X, y, p_set.tile_size)

                # Save run information and results.
                results = OrderedDict()
                results["descriptor"] = str(descriptor)
                results["radius"] = p_set.scale[1]
                results["num points"] = p_set.scale[0]
                results["distance metric"] = p_set.distance_metric
                results["histograms"] = p_set.to_hist
                results["tile size"] = p_set.tile_size
                results["rank1"] = acc

                run_data.append(results)

            run_data_df = pd.DataFrame.from_dict(run_data, orient="columns")
            run_data_df = run_data_df.sort_values("rank1", ascending=False)
            print(run_data_df)

            run_data_df.to_csv(results_dir + f"run_{str(run_time).replace(' ', '_')}", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Specify task number [1 - 5]")
        exit()

    TaskRunner.run(int(sys.argv[1]))
