# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:39:54 2021

@author: aow001
"""

# ===========================================================================
# Name        : main_volta.py
# Author      : aow001, adapted from MarkW  (adapted JazminZ & MatteoG)
# Version     : 0.05
# Copyright   : Your copyright notice
# ===========================================================================
import csv
import logging
import numpy as np
import os
import pandas as pd
import random

#pip install platypus
from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator

import rbf_functions
from Volta_model_2 import VoltaModel


class TrackProgress:
    def __init__(self):
        self.nfe = []
        self.improvements = []
        self.objectives = {}

    def __call__(self, algorithm):
        self.nfe.append(algorithm.nfe)
        self.improvements.append(algorithm.archive.improvements)
        temp = {}
        for i, solution in enumerate(algorithm.archive):
            temp[i] = list(solution.objectives)
        self.objectives[algorithm.nfe] = pd.DataFrame.from_dict(temp,
                                                                orient='index')

    def to_dataframe(self):
        df_imp = pd.DataFrame.from_dict(dict(nfe=self.nfe,
                                             improvements=self.improvements))
        df_hv = pd.concat(self.objectives, axis=0)
        return df_imp, df_hv


def store_results(algorithm, track_progress, output_dir, rbf_name,
                  seed_id):
    path_name = f"{output_dir}/{rbf_name}"
    if not os.path.exists(path_name):
        try:
            os.mkdir(path_name)
        except OSError:
            print("Creation of the directory failed")


    header = ["annual_hydropower", "daily_hydropower", "irrigation",
              "environment", "flood_control"]
    with open(f"{output_dir}/{rbf_name}/{seed_id}_solution.csv", "w",
              encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for solution in algorithm.result:
            writer.writerow(solution.objectives)

    with open(f"{output_dir}/{rbf_name}/{seed_id}_variables.csv", "w",
              encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        for solution in algorithm.result:
            writer.writerow(solution.variables)

    # save progress info
    df_conv, df_hv = track_progress.to_dataframe()
    df_conv.to_csv(f"{output_dir}/{rbf_name}/{seed_id}_convergence.csv")
    df_hv.to_csv(f"{output_dir}/{rbf_name}/{seed_id}_hypervolume.csv")


def main():
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for entry in [
                  #rbf_functions.original_rbf,
        rbf_functions.squared_exponential_rbf,
        rbf_functions.inverse_multiquadric_rbf,
        #rbf_functions.inverse_quadratic_rbf,
        #rbf_functions.exponential_rbf,
        rbf_functions.matern32_rbf,
        #rbf_functions.matern52_rbf,
   ]:
        for seed in seeds:
            random.seed(seed)

            # RBF parameters
            n_inputs = 2  # (time, storage of Akosombo)
            n_outputs = 3    #(irrigation and downstream release but with floods separated)
            n_rbfs = 4      # how is this determined
            rbf = rbf_functions.RBF(n_rbfs, n_inputs, n_outputs, rbf_function=entry)

            # Initialize model
            n_objectives = 5
            n_years = 1

            lowervolta_river = VoltaModel(265, 505, n_years, rbf)
            lowervolta_river.set_log(False)

            # Lower and Upper Bound for problem.types
            epsilons = [0.5, 0.05, 0.05, 0.05, 0.001, 0.05]     #are these for each objective? 
            n_decision_vars = len(rbf.platypus_types)

            problem = Problem(n_decision_vars, n_objectives)
            problem.types[:] = rbf.platypus_types
            problem.function = lowervolta_river.evaluate

            problem.directions[0] = Problem.MAXIMIZE  # annual hydropower #***I see that all of these are maximize even though the env function should be minimized...
            problem.directions[1] = Problem.MAXIMIZE  # daily hydropower
            problem.directions[2] = Problem.MAXIMIZE  # irrigation
            problem.directions[3] = Problem.MAXIMIZE  # environment
            problem.directions[4] = Problem.MINIMIZE  # flood events

            # algorithm = EpsNSGAII(problem, epsilons=epsilons)
            # algorithm.run(1000)

            track_progress = TrackProgress()
            with ProcessPoolEvaluator() as evaluator:
                algorithm = EpsNSGAII(problem, epsilons=epsilons,
                                      evaluator=evaluator)
                algorithm.run(500, track_progress) #original was 50000

            store_results(algorithm, track_progress, 'output',
                          f"{entry.__name__}",
                          seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
