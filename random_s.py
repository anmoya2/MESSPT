from ast import operator
import collections
import copy
import math
import numbers
import random
import typing

import numpy as np
import time

from river import anomaly, base, compose, drift, metrics, utils


ModelWrapper = collections.namedtuple("ModelWrapper", "estimator metric")



class RANDOM(base.Estimator):
    """Single-pass Self Parameter Tuning

    Parameters
    ----------
    estimator
    metric
    params_range
    drift_input
    grace_period
    drift_detector
    convergence_sphere
    seed
        
    n_ind: int,
    len_ind: int
    gen_to_conv: int
	self,
    
    """

    _START_RANDOM = "random"
    _START_WARM = "warm"

    def __init__(
        self,
        estimator: base.Estimator,
        metric: metrics.base.Metric,
        params_range: typing.Dict[str, typing.Tuple],
        drift_input: typing.Callable[[float, float], float],
        grace_period: int = 500,
        drift_detector: base.DriftDetector = drift.ADWIN(),
        convergence_sphere: float = 0.001,
        seed: int = None,
        #n_gen: borrado
        #n_ind: 4--> cambio a 8
        n_ind: int = 8,
        len_ind: int = 3,
        gen_to_conv: int = 4

        #m_prob borrado

    ):
        super().__init__()
        ##to measure time and cost
        self._time_to_conv = []
        self._time_acc = 0
        self._num_ex = 0
        self._num_ex_array = []
        self._num_models = 0
        self._num_models_array = []
        ##
        self.estimator = estimator
        self.metric = metric
        self.params_range = params_range
        self.drift_input = drift_input
        self._converged = False
        self.grace_period = grace_period
        self.drift_detector = drift_detector
        self.convergence_sphere = convergence_sphere
        self.seed = seed
        self._rng = random.Random(self.seed)
        self._n = 0
        self.len_ind = len_ind
        self.n_ind = n_ind
        #self._first_c = True
        self._old_b = None
        #
        self._gen_to_conv = gen_to_conv
        self.current_pop = []
        self._best_c = None
        
        #c_best
        self._best_estimator = None
        self._best_i = None
        self._first_pass = True
        self._gen_same_b = 0
        self._create_first(self.n_ind)



    def __generate(self, p_data):
        p_type, p_range = p_data
        if p_type == int:
            return self._rng.randint(p_range[0], p_range[1])
        elif p_type == float:
            return self._rng.uniform(p_range[0], p_range[1])


    def _random_config(self):
        return self._recurse_params(
            operation = "generate",
            p_data=self.params_range,
            e1_data=self.estimator._get_params()
        )


    def _create_wrapper(self, ind, metric):
        w = ModelWrapper(self.estimator.clone(ind, include_attributes=True), metric)
        return w

    
    def _create_first(self, n):
        self.current_pop = [None]*n
        for i in range(n):
            self.current_pop[i] = self._create_wrapper(self._random_config(), 
                self.metric.clone(include_attributes=True))
    

    def _normalize_flattened_hyperspace(self, orig):
        scaled = {}
        self._recurse_params(
            operation="scale",
            p_data=self.params_range,
            e1_data=orig,
            prefix="",
            scaled=scaled
        )
        return scaled

    @property
    def best(self):
        
        
        return self._best_estimator

    @property
    def _models_converged(self) -> bool:
        # Normalize params to ensure they contribute equally to the stopping criterion
        
        scaled_params_b = self._normalize_flattened_hyperspace(
                self._best_i.estimator._get_params()
            )
        r_sphere=1
        
        if self._old_b != None:
            r_sphere = utils.math.minkowski_distance(scaled_params_b, 
                self._old_b, p=2)

        #check how many gen best has small changes
        if r_sphere < self.convergence_sphere:
            if self._gen_same_b == self._gen_to_conv:
                self._gen_same_b = 0
                self._old_b = None
                return True
            else:
                self._gen_same_b = self._gen_same_b+1
                return False
        else:
            self._gen_same_b = 0
            self._old_b = scaled_params_b
            return False

    

    def __combine(self, p_info, param1, param2, index_change, gen, func):

        p_type, p_range = p_info
        new_val = func(param1, param2, index_change, gen)

        # Range sanity checks
        if new_val < p_range[0]:
            new_val = p_range[0]
        if new_val > p_range[1]:
            new_val = p_range[1]

        new_val = round(new_val, 0) if p_type == int else new_val
        return new_val

    def _gen_new_estimator(self, e1, index_change, func):
        """Generate new configuration given two estimators and a combination function."""

        e1_p = e1.estimator._get_params()

        new_config = self._recurse_params(
            operation="combine",
            p_data=self.params_range,
            e1_data=e1_p,
            index_change=index_change,
            func=func,
            gen=0
        )
        # Modify the current best contender with the new hyperparameter values
        new = ModelWrapper(
            copy.deepcopy(self._best_i.estimator),
            self.metric.clone(include_attributes=True),
        )
        new.estimator.mutate(new_config)

        return new


    def __flatten(self, prefix, scaled, p_info, e_info):
        _, p_range = p_info
        interval = p_range[1] - p_range[0]
        scaled[prefix] = (e_info - p_range[0]) / interval


    def _recurse_params(
        self, operation, p_data, e1_data, *, index_change=None, 
        func=None, e2_data=None,
         prefix=None, scaled=None, gen=None
    ):  

        # Sub-component needs to be instantiated
        if isinstance(e1_data, tuple):
            sub_class, sub_data1 = e1_data

            
            sub_data2 = {}


            sub_config = {}
            for sub_param, sub_info in p_data.items():
                if operation == "scale":
                    sub_prefix = prefix + "__" + sub_param
                else:
                    sub_prefix = None
                sub_config[sub_param] = self._recurse_params(
                    operation=operation,
                    p_data=sub_info,
                    e1_data=sub_data1[sub_param],
                    func=func,
                    e2_data=sub_data2.get(sub_param, None),
                    index_change=index_change,
                    prefix=sub_prefix,
                    scaled=scaled,
                    gen=gen
                )
            return sub_class(**sub_config)

        # We reached the numeric parameters
        if isinstance(p_data, tuple):
            if operation == "generate":
                return self.__generate(p_data)
            if operation == "scale":
                self.__flatten(prefix, scaled, p_data, e1_data)
                return
           
            # combine

            new_v = self.__generate(p_data)

            res = self.__combine(p_data, e1_data, new_v, 
                    index_change, gen, func)
            return res


        # The sub-parameters need to be expanded
        config = {}
        for p_name, p_info in p_data.items():
            e1_info = e1_data[p_name]

            
            e2_info = {}

            if operation == "scale":
                sub_prefix = prefix + "__" + p_name if len(prefix) > 0 else p_name
            else:
                sub_prefix = None

            if not isinstance(p_info, dict):
                config[p_name] = self._recurse_params(
                    operation=operation,
                    p_data=p_info,
                    e1_data=e1_info,
                    func=func,
                    e2_data=e2_info,
                    index_change=index_change,
                    prefix=sub_prefix,
                    scaled=scaled,
                    gen=gen
                )
            else:
                sub_config = {}
                for sub_name, sub_info in p_info.items():
                    if operation == "scale":
                        sub_prefix2 = sub_prefix + "__" + sub_name
                    else:
                        sub_prefix2 = None
                    sub_config[sub_name] = self._recurse_params(
                        operation=operation,
                        p_data=sub_info,
                        e1_data=e1_info[sub_name],
                        func=func,
                        e2_data=e2_info.get(sub_name, None),
                        index_change=index_change,
                        prefix=sub_prefix2,
                        scaled=scaled,
                        gen=gen
                    )
                    if operation == "combine":
                        gen = gen+1
                config[p_name] = sub_config
        return config

    
    def _func_cr(self, p1, p2, index_change, gen):
                
        if index_change!=gen:
            return p1
        else:
            return p2

    def _create_new_pop(self):

        for i in range(self.n_ind):
            current_el = self.current_pop[i]
            
            index_change = self._rng.randint(0, self.len_ind-1)
            
            new_c = self._gen_new_estimator(current_el,
                index_change, self._func_cr)
            self.current_pop[i] = new_c

    def _sort_c(self):
        """Ensure the simplex models are ordered by predictive performance."""
        if self.metric.bigger_is_better:
            self.current_pop.sort(key=lambda mw: mw.metric.get(), reverse=True)
        else:
            self.current_pop.sort(key=lambda mw: mw.metric.get())
        

    def _learn_one(self, wrap, x, y):
        wrap.estimator.learn_one(x,y)

    def _update(self, wrap, x, y):
        scorer = getattr(wrap.estimator, "predict_one")
        y_pred = scorer(x)
        
        wrap.metric.update(y, y_pred)
    
           
    def _learn_converged(self, x, y):
        scorer = getattr(self._best_estimator, "predict_one")
        y_pred = scorer(x)

        input = self.drift_input(y, y_pred)
        self.drift_detector.update(input)

        # We need to start the optimization process from scratch
        if self.drift_detector.drift_detected:
            print("drift detected")
            self._n = 0
            self._converged = False
            #keep the best
            self.current_pop[0] = ModelWrapper(copy.deepcopy(self._best_estimator),
                #self._best_i.metric.clone(include_attributes=True)
                self.metric.clone(include_attributes=True))
            for i in range(1, self.n_ind):
                self.current_pop[i] = self._create_wrapper(self._random_config(), 
                    self.metric.clone(include_attributes=True))

            # There is no proven best model right now. Restart
            self._best_i = None
            self._best_estimator = None
            self._time_acc = 0
            self._num_ex = 0
            self._num_models = 0
            self._first_pass = True
            return

        self._best_estimator.learn_one(x, y)

    def learn_one(self, x, y):
        
        if self._converged ==True:
            self._n = self._n + 1
            self._learn_converged(x, y)
        else:
            t1 = time.time()
            
            self._num_ex = self._num_ex +1
            #actualize individuals of pop
            for wrap in self.current_pop:

                self._update(wrap, x, y)
                self._learn_one(wrap, x, y)



            self._sort_c()
            
            if self._first_pass:
                self._best_i = self.current_pop[0]
                

            else:
                self._update(self._best_c, x, y)
                self._learn_one(self._best_c, x, y)
                if self.metric.bigger_is_better:
                    if self.current_pop[0].metric.get() > self._best_c.metric.get():
                        self._best_i = self.current_pop[0]
                    else:
                        self._best_i = self._best_c
                else:
                    if self.current_pop[0].metric.get() < self._best_c.metric.get():
                        self._best_i = self.current_pop[0]
                    else:
                        self._best_i = self._best_c
                    
                    
            
            self._best_estimator = self._best_i.estimator

            
            
            self._n = self._n + 1
            #Actualize pop
            if self._n % self.grace_period == 0:
                self._num_models = self._num_models + self.n_ind+1
                if self._first_pass:
                    self._best_c = self.current_pop[0]
                                
                else:
                    if self.metric.bigger_is_better:
                        if self.current_pop[0].metric.get() > self._best_c.metric.get():
                            self._best_c = self.current_pop[0]
                    else:
                        if self.current_pop[0].metric.get() < self._best_c.metric.get():
                            self._best_c = self.current_pop[0]
                    
                

                if self._models_converged:
                    t2 = time.time()
                    self._time_acc = self._time_acc + (t2-t1)

                    self._time_to_conv.append(self._time_acc)
                    self._num_ex_array.append(self._num_ex)
                    self._num_models_array.append(self._num_models)

                    self._converged = True
                else:
                    self._best_c = ModelWrapper(copy.deepcopy(self._best_c.estimator),
                        
                        self.metric.clone(include_attributes=True))    
                    self._create_new_pop()
                            
                
            t2 = time.time()
            self._time_acc = self._time_acc + (t2-t1)
            
        return self 


    def predict_one(self, x, **kwargs):
        if self._best_estimator == None:
            self._sort_c()
            return self.current_pop[0].estimator.predict_one(x, **kwargs)
        
        return self._best_estimator.predict_one(x, **kwargs)

    @property
    def converged(self):
        return self._converged
        
    @property
    def time_to_conv(self):
        if self.converged:
            return self._time_to_conv
        else:
            self._time_to_conv.append(self._time_acc)
            return self._time_to_conv
    
    @property
    def num_ex(self):
        if self.converged:
            return self._num_ex_array
        else:
            self._num_ex_array.append(self._num_ex)
            return self._num_ex_array

    @property
    def num_models(self):
        if self.converged:
            return self._num_models_array
        else:
            self._num_models_array.append(self._num_models)
            return self._num_models_array
        