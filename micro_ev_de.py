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



class MICRODE(base.Estimator):
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
    n_ind
    reset_one
    len_ind
    gen_to_conv
    F_ini
    CR_ini
    aug
    seed

    

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
        n_ind: int = 4,
        reset_one: int = 0,
        len_ind: int = 3,
        gen_to_conv: int = 5,
        F_ini: float = 0.5,
        CR_ini: float= 0.5,
        aug: float = 0.025,

    ):
        super().__init__()

        self.random_option = None
        self._old_b_rand_opt = None
        self.reset_one = reset_one
        ##Measure cost
        self._time_to_conv = []
        self._time_acc = 0
        self._num_ex = 0
        self._num_ex_array = []
        self._num_models = 0
        self._num_models_array = []
        ##

        self.len_ind = len_ind
        self.estimator = estimator
        self.metric = metric
        self.params_range = params_range
        self.drift_input = drift_input
        self._converged = False
        self.grace_period = grace_period
        self.drift_detector = drift_detector
        self.convergence_sphere = convergence_sphere
        #self.first_pop = True
        self.seed = seed
        self._rng = random.Random(self.seed)
        self._n = 0
        self._old_b = None
        #number of generations (max)
        self._gen_to_conv = gen_to_conv
        #number of individuals into pop
        self.n_ind = n_ind
        #probability of mutation
        self.current_pop = []

        self._best_i = None
        self._best_child = False
        
        self.current_children = []
        
        
        #c_best
        self._random_best = False
        self._best_estimator = None
        self._gen_same_b = 0
        self.is_adaptive = True
        self.F_ini = F_ini
        self.CR_ini = CR_ini
        self.F = self.F_ini 
        self.CR = self.CR_ini 

        self.aug = aug
        self._create_pop(self.n_ind)


    def __generate(self, p_data):
        p_type, p_range = p_data
        if p_type == int:
            return self._rng.randint(p_range[0], p_range[1])
        elif p_type == float:
            return self._rng.uniform(p_range[0], p_range[1])
        #DISCRET
        ''' 
        elif p_type == "discrete":
            choice = self._rng.randint(0, len(p_range))
            return p_range[choice]
        '''

    def _random_config(self):
        return self._recurse_params(
            operation = "generate",
            p_data=self.params_range,
            e1_data=self.estimator._get_params()
        )


    def _create_wrapper(self, ind, metric, est):
        w = ModelWrapper(est.clone(ind, include_attributes=True), metric)
        return w


    def _create_pop(self, n):
        self.current_pop = [None]*n
        for i in range(n):
            self.current_pop[i] = self._create_wrapper(self._random_config(), 
                self.metric.clone(include_attributes=True), self.estimator)

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
                self._best_estimator._get_params()
            )
        r_sphere=1
        
        if self._old_b != None:
            r_sphere = utils.math.minkowski_distance(scaled_params_b, 
                self._old_b, p=2)


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


    def __combine(self, p_info, param1, param2, param3, func):

        p_type, p_range = p_info
        new_val = func(param1, param2, param3)

        # Range sanity checks
        if new_val < p_range[0]:
            new_val = p_range[0]
        if new_val > p_range[1]:
            new_val = p_range[1]

        new_val = round(new_val, 0) if p_type == int else new_val
        return new_val

    def _gen_new_estimator(self, e1, e2, e3, func):
        """Generate new configuration given two estimators and a combination function."""
        e1_p = e1.estimator._get_params()
        e2_p = e2.estimator._get_params()
        e3_p = e3.estimator._get_params()


        new_config = self._recurse_params(
            operation="combine",
            p_data=self.params_range,
            e1_data=e1_p,
            func=func,
            e2_data=e2_p,
            e3_data=e3_p,
        )
        # Modify the current best contender with the new hyperparameter values
        new = ModelWrapper(
            copy.deepcopy(self._best_estimator),
            self.metric.clone(include_attributes=True),

        )
        
        new.estimator.mutate(new_config)

        return new

    def __combine_cross_rate(self, p_info, param1, param2, num, index_change, gen, func):

        p_type, p_range = p_info
        new_val = func(param1, param2, num, index_change, gen)

        # Range sanity checks
        if new_val < p_range[0]:
            new_val = p_range[0]
        if new_val > p_range[1]:
            new_val = p_range[1]

        new_val = round(new_val, 0) if p_type == int else new_val
        return new_val

    def _gen_new_estimator_cross_rate(self, e1, e2, index_change, func):
        """Generate new configuration given two estimators and a combination function."""

        e1_p = e1.estimator._get_params()
        e2_p = e2.estimator._get_params()


        new_config = self._recurse_params(
            operation="combine_cr",
            p_data=self.params_range,
            e1_data=e1_p,
            index_change=index_change,
            func=func,
            e2_data=e2_p,
            gen=0
        )
        # Modify the current best contender with the new hyperparameter values
        new = ModelWrapper(
            copy.deepcopy(e2.estimator),
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
        func=None, e2_data=None, e3_data=None,
         prefix=None, scaled=None, gen=None
    ):  

        # Sub-component needs to be instantiated
        if isinstance(e1_data, tuple):
            sub_class, sub_data1 = e1_data

            if operation=="combine_cr":
                _, sub_data2 = e2_data
                sub_data3 = {}

            elif operation == "combine":
                _, sub_data2 = e2_data
                _, sub_data3 = e3_data

            else:
                sub_data2 = {}
                sub_data3 = {}


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
                    e3_data=sub_data3.get(sub_param, None),
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
            if operation == "combine_cr":
                num = self._rng.uniform(0,1)
                res = self.__combine_cross_rate(p_data, e1_data, e2_data, num, 
                    index_change, gen, func)
                return res

            # combine
            #NEW
            p_type, p_range = p_data
            if p_type == int or p_type == float:
                return self.__combine(p_data, e1_data, e2_data, e3_data, func)
            else:
                return self.__combine(p_data, e1_data, e2_data, e3_data, func_disc)
        # The sub-parameters need to be expanded
        config = {}
        for p_name, p_info in p_data.items():
            e1_info = e1_data[p_name]

            if operation == "combine":
                e2_info = e2_data[p_name]
                e3_info = e3_data[p_name]

            elif operation == "combine_cr":
                e2_info = e2_data[p_name]
                e3_info = {}

            else:
                e2_info = {}
                e3_info = {}

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
                    e3_data=e3_info,
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
                        e3_data=e3_info.get(sub_name, None),
                        index_change=index_change,
                        prefix=sub_prefix2,
                        scaled=scaled,
                        gen=gen
                    )
                    if operation == "combine_cr":
                        gen = gen+1
                config[p_name] = sub_config
        return config

    def func_disc(self, v1, v2, v3):
        r_f_c = self._rng.uniform(0,1)
        to_c = v2 if r_f_c <= 0.5 else v3
        r_f_c_2 = self._rng.uniform(0,1)
        res = to_c if r_f_c_2 < self.F/(1+self.F) else v1
        return res





    def _de_cross_best_1(self, i):
        list_el = np.ndarray.tolist(np.arange(1,self.n_ind))
        if i!=0:
            list_el.remove(i)
        r1, r2 = self._rng.sample(list_el, 2)
        #best in 0 pos
        n_p = self._gen_new_estimator(
            self.current_pop[0], self.current_pop[r1], self.current_pop[r2],
             lambda h1, h2, h3: h1 + self.F*(h2-h3)
        )
        return n_p

    def _ev_op_crossover(
        self, operation, ind=None, index=None):
        
        return self._de_cross_best_1(index)


    def func_cr(self, p1, p2, num, index_change, gen):
                
        if num<self.CR or index_change==gen:
            return p1
        else:
            return p2


    def _cross_current_pop(self):
        for i in range(self.n_ind):
            current_el = self.current_pop[i]
            new_c = self._ev_op_crossover("de_best_1", None, i)
            index_change = self._rng.randint(0, self.len_ind-1)
            
            new_c = self._gen_new_estimator_cross_rate(new_c, current_el,
                index_change, self.func_cr)
            self.current_children.append(new_c)
    

    def _learn_one(self, wrap, x, y):
        wrap.estimator.learn_one(x,y)

    def _update(self, wrap, x, y):
        scorer = getattr(wrap.estimator, "predict_one")


        y_pred = scorer(x)
        
        wrap.metric.update(y, y_pred)

    def _sort_c_pop(self):
        """Ensure the simplex models are ordered by predictive performance."""
        if self.metric.bigger_is_better:
            self.current_pop.sort(key=lambda mw: mw.metric.get(), reverse=True)
        else:
            self.current_pop.sort(key=lambda mw: mw.metric.get())

    def _sort_c_children(self):
        """Ensure the simplex models are ordered by predictive performance."""
        if self.metric.bigger_is_better:
            self.current_children.sort(key=lambda mw: mw.metric.get(), reverse=True)
        else:
            self.current_children.sort(key=lambda mw: mw.metric.get())

    def _generate_next_current_pop(self):
        
        
        for i in range(self.n_ind):
            
            self.current_pop[i] = ModelWrapper(copy.deepcopy(self._best_i.estimator),
            self.metric.clone(include_attributes=True)

            )
            if i!=0:
                self.current_pop[i].estimator.mutate(self._random_config())
            
        
        if self.reset_one==1:
            self.random_option = self._create_wrapper(self._random_config(), 
                    self.metric.clone(include_attributes=True), self.estimator)




           
    def _learn_converged(self, x, y):
        scorer = getattr(self._best_estimator, "predict_one")
        y_pred = scorer(x)

        input = self.drift_input(y, y_pred)
        self.drift_detector.update(input)

        # We need to start the optimization process from scratch
        if self.drift_detector.drift_detected:
            self.F = self.F_ini
            self.CR = self.CR_ini
            if self.reset_one==1:
                print("drift detected mder")
            else:
                print("drift detected mde")

            self._n = 0
            self._converged = False
            self._best_child = False
            
            for i in range(1, self.n_ind):
                self.current_pop[i] = self._create_wrapper(self._random_config(), 
                    #self._best_i.metric.clone(include_attributes=True)
                    self.metric.clone(include_attributes=True), self.estimator

                )
            
            self.current_pop[0] = ModelWrapper(copy.deepcopy(self._best_estimator),
                self.metric.clone(include_attributes=True))
            



            # There is no proven best model right now
            self._best_i = None
            self._best_estimator = None
            self.random_option = None
            self._random_best = None
            self._time_acc = 0
            self._num_ex = 0
            self._num_models = 0

            return

        self._best_estimator.learn_one(x, y)

    def learn_one(self, x, y):
        

        if self._converged ==True:
            self._n = self._n + 1
            self._learn_converged(x, y)
        else:
            #num ex not converged  
            t1 = time.time()
            self._num_ex = self._num_ex + 1
            
            #actualize individuals of pop
            for wrap in self.current_pop:
                self._update(wrap, x, y)

                self._learn_one(wrap, x, y)

            self._sort_c_pop()


            if self.current_children != []:
                for wrap in self.current_children:
                    self._update(wrap, x, y)
                    self._learn_one(wrap, x, y)
                self._sort_c_children()

            if self.current_children != []:
                if self.metric.bigger_is_better:
                    if self.current_pop[0].metric.get() < self.current_children[0].metric.get():
                        self._best_i = self.current_children[0]
                        self._best_child = True
                    else:
                        self._best_i = self.current_pop[0]
                        self._best_child = False
                else:
                    if self.current_pop[0].metric.get() > self.current_children[0].metric.get():
                        self._best_i = self.current_children[0]
                        self._best_child = True
                    else:
                        self._best_i = self.current_pop[0]
                        self._best_child = False
            else:
                self._best_i = self.current_pop[0]
                self._best_child = False


            self._random_best = False
            if self.random_option != None:
                self._update(self.random_option, x, y)
                self._learn_one(self.random_option, x, y)
                if self.metric.bigger_is_better:
                    if self.random_option.metric.get() > self._best_i.metric.get():

                        self._old_b_rand_opt = self._best_i
                        
                        self._best_i = self.random_option

                        self._random_best = True
                else:
                    if self.random_option.metric.get() < self._best_i.metric.get():
                        self._old_b_rand_opt = self._best_i
                        self._best_i = self.random_option
                        self._random_best = True


            
            self._best_estimator = self._best_i.estimator

            self._n = self._n + 1
            #Actualize pop
            if self._n % self.grace_period == 0:
                if self._random_best == True:
                    self.F = self.F_ini
                    self.CR = self.CR_ini
                    self._n = 0
                    self._converged = False
                    self._best_child = False
                    self._random_best = False
                    self.random_option = None
                    
                    for i in range(2, self.n_ind):
                        self.current_pop[i] = self._create_wrapper(self._random_config(), 
                            self.metric.clone(include_attributes=True), self.estimator

                        )
                    
                    self.current_pop[0] = ModelWrapper(copy.deepcopy(self._best_estimator),
                        self.metric.clone(include_attributes=True))
                
                    #copy best before rand
                    self.current_pop[1] = ModelWrapper(copy.deepcopy(self._old_b_rand_opt.estimator),
                        self.metric.clone(include_attributes=True))

                    self._old_b_rand_opt = None
                    self.current_children = []


                    # There is no proven best model right now
                    self._best_i = None
                    self._best_estimator = None
                #first grace period
                elif self._n / self.grace_period == 1:
                    self._num_models = self._num_models + self.n_ind
                    
                    #firstly, models are mixed
                    self._cross_current_pop()
                    
                    for j in range(self.n_ind):
                        self.current_pop[j] = ModelWrapper(
                        copy.deepcopy(self.current_pop[j].estimator),
                            self.metric.clone(include_attributes=True),

                            )

                else:
                    self._num_models = self._num_models + self.n_ind*2+1

                    if self.is_adaptive:
                        #to fast conv
                        self.F = self.F - self.aug
                        self.CR = self.CR + self.aug
                        if self.F < 0.0:
                            self.F = 0.0
                        if self.CR > 1:
                            self.CR = 1
                    
                    self._generate_next_current_pop()
                    self.current_children=[]
                    self._best_child = False
                    
                    if self._models_converged:
                        t2 = time.time()
                        if self.reset_one==0:
                            print("mde convergence")
                        else:
                            print("mder convergence")
                        self._time_acc = self._time_acc + (t2-t1)
                        self._time_to_conv.append(self._time_acc)
                        self._num_ex_array.append(self._num_ex)
                        self._num_models_array.append(self._num_models)
                        self._converged = True
                    else:
                        self._cross_current_pop()
                                        
                

            t2 = time.time()
            self._time_acc = self._time_acc + (t2-t1)
        return self 


    def predict_one(self, x, **kwargs):
        if self._best_estimator == None:
            self._sort_c_pop()
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