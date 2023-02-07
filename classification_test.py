from river import datasets, drift, linear_model, metrics, preprocessing, tree, utils

from sspt import SSPT
from micro_ev_de import MICRODE
from random_s import RANDOM
import time
import csv
import random
import numpy as np



class SEA(datasets.base.SyntheticDataset):

    def __init__(self, variant=0, noise=0.0, seed: int = None):

        super().__init__(n_features=3, task=datasets.base.BINARY_CLF)

        if variant not in (0, 1, 2, 3):
            raise ValueError("Unknown variant, possible choices are: 0, 1, 2, 3")

        self.variant = variant
        self.noise = noise
        self.seed = seed
        self._threshold = {0: 6, 1: 14, 2: 3, 3: 9}[variant]

    def __iter__(self):

        rng = random.Random(self.seed)

        while True:

            x = {i: rng.uniform(0, 10) for i in range(3)}
            y = x[0] + x[1] > self._threshold

            if self.noise and rng.random() < self.noise:
                y = not y

            yield x, y

    @property
    def _repr_content(self):
        return {**super()._repr_content, "Variant": str(self.variant)}

from scipy.io import arff
import pandas as pd

from random import shuffle

gp = 500
rolling_wind = 1000

seeds = [111, 222, 333, 444, 555, 666]
n_exec = 0

dataset_name = "SEA"

## Introduce here the dataset

dataset = datasets.synth.ConceptDriftStream(
    stream=SEA(seed=42, variant=0),
    drift_stream=SEA(seed=42, variant=1),
    seed=seeds[n_exec],
    position=25000,
    width=2,
).take(50000)



# SSPT - model and metric
sspt_metric = metrics.Accuracy()
sspt_rolling_metric = utils.Rolling(metrics.Accuracy(), window_size=rolling_wind)
#case you want to plt
sspt_metric_plt = []
sspt_rolling_metric_plt = []

# Random - model and metric
random_metric = metrics.Accuracy()
random_rolling_metric = utils.Rolling(metrics.Accuracy(), window_size=rolling_wind)
#case you want to plt
random_metric_plt = []
random_rolling_metric_plt = []

# MESSPT - model and metric
mder_metric = metrics.Accuracy()
mder_rolling_metric = utils.Rolling(metrics.Accuracy(), window_size=rolling_wind)
#case you want to plt
mder_metric_plt = []
mder_rolling_metric_plt = []



#SSPT
sspt = SSPT(
    estimator=preprocessing.AdaptiveStandardScaler() | tree.HoeffdingTreeClassifier(),
    metric=sspt_metric,
    grace_period=gp,
    params_range={
        "HoeffdingTreeClassifier": {
            "delta": (float, (0.00001, 0.0001)),
            "grace_period": (int, (100, 500)),
            "tau": (float, (0.01, 0.09)),

        },
    },
    drift_input=lambda yt, yp: 0 if yt == yp else 1,
    convergence_sphere=0.000001,
    seed=seeds[n_exec],
)

#random search
rnd = RANDOM(
    estimator=preprocessing.AdaptiveStandardScaler() | tree.HoeffdingTreeClassifier(),
    metric=random_metric,
    grace_period=gp,
    params_range={
        "HoeffdingTreeClassifier": {
            "delta": (float, (0.00001, 0.0001)),
            "grace_period": (int, (100, 500)),
            "tau": (float, (0.01, 0.09)),

        },
    },
    drift_input=lambda yt, yp: 0 if yt == yp else 1,
    convergence_sphere=0.000001,
    seed=seeds[n_exec],
)

#MESSPT
mder = MICRODE(
    reset_one=1,
    estimator=preprocessing.AdaptiveStandardScaler() | tree.HoeffdingTreeClassifier(),
    metric=mder_metric,
    grace_period=gp,
    params_range={
        "HoeffdingTreeClassifier": {
            "delta": (float, (0.00001, 0.0001)),
            "grace_period": (int, (100, 500)),
            "tau": (float, (0.01, 0.09)),

        },
    },
    drift_input=lambda yt, yp: 0 if yt == yp else 1,
    convergence_sphere=0.0005,
    seed=seeds[n_exec],
)



first_print = True
first_print_r = True
first_print_m = True
first_print_d = True
first_print_mder = True
first_print_der = True

converged_yet = False
converged_yet_rnd = False
converged_yet_de = False
converged_yet_mde = False
converged_yet_der = False
converged_yet_mder = False

#Iterate over dataset. x has to be a dict.

for i, (x,y) in enumerate(dataset):
    
    
    sspt_y_pred = sspt.predict_one(x)
    sspt_metric.update(y, sspt_y_pred)
    sspt_rolling_metric.update(y, sspt_y_pred)
    sspt_metric_plt.append(sspt_metric.get())
    sspt_rolling_metric_plt.append(sspt_rolling_metric.get())
    sspt.learn_one(x, y)

    random_y_pred = rnd.predict_one(x)
    
    random_metric.update(y, random_y_pred)
    random_rolling_metric.update(y, random_y_pred)
    random_metric_plt.append(random_metric.get())
    random_rolling_metric_plt.append(random_rolling_metric.get())
    rnd.learn_one(x, y)
    

    mder_y_pred = mder.predict_one(x)

    
    mder_metric.update(y, mder_y_pred)
    mder_rolling_metric.update(y, mder_y_pred)
    mder_metric_plt.append(mder_metric.get())



    mder_rolling_metric_plt.append(mder_rolling_metric.get())
    



    mder.learn_one(x, y)
    
    
    
    if sspt.converged and first_print:
        print("Converged sspt at:", i)
        first_print = False
        converged_yet = True

    if rnd.converged and first_print_r:
        print("Converged random at:", i)
        first_print_r = False
        converged_yet_rnd = True
    

    if mder.converged and first_print_mder:
        print("Converged mder at:", i)
        first_print_mder = False
        converged_yet_mder = True


    

    if sspt.converged == False and converged_yet==True:
        print("Drift sspt at:", i)
        first_print = True
        converged_yet = False

    if rnd.converged == False and converged_yet_rnd ==True:
        print("Drift rnd at:", i)
        first_print_r = True
        converged_yet_rnd = False
    
    

    if mder.converged == False and converged_yet_mder ==True:
        print("Drift mder at:", i)
        first_print_mder = True
        converged_yet_mder = False
    



print("Total instances:", i + 1)

print("Best params sspt:")
print(repr(sspt.best))
print("Best params random:")
print(repr(rnd.best))

print("Best params mder:")
print(repr(mder.best))


print("SSPT: ", sspt_metric)
print("random: ", random_metric)
print("mder: ", mder_metric)

#Util information regarding each algorithm

print("RANDOM RESULTS: ")
print("random num ex: ", rnd.num_ex)
print("random num models: ", rnd.num_models )
print("random time to conv: ", rnd.time_to_conv)

print("SSPT RESULTS: ")
print("sspt num ex: ", sspt.num_ex)
print("sspt num models: ", sspt.num_models)
print("sspt time to conv: ", sspt.time_to_conv)

print("MDER RESULTS: ")
print("mder num ex: ", mder.num_ex)
print("mde num models: ", mder.num_models)
print("mder time to conv: ", mder.time_to_conv)

