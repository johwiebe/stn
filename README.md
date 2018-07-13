# STN with degradation

Implementation of the State-Task-Network (STN) [Kondili et al. 1993] with
degradation of equipment. This code can be used to replicate the results in [Wiebe et al. 2018].

## Credit
This implementation is based on the [STN-Scheduler](https://github.com/jckantor/STN-Scheduler) by Jeffrey Kantor (c) 2017.

## Dependencies

* [Pyomo](http://www.pyomo.org/)
* A MILP solver. The module has been tested with
  [CPLEX](https://www.ibm.com/analytics/cplex-optimizer).

## Usage

#### lhs.py
Generate data for logistic regression by solving short-term scheduling model
repeatedly for different demands:
```
python lhs.py runs/test_lhs.yaml
```
where `runs/test_lhs.yaml` is a [config file]().

#### logreg.py
Train logistic regression for Markov-chain or frequency approach.

#### rolling.py
Solve model using rolling horizon:
```
python rolling runs/test_det.yaml
```

#### mc.py
Solve model using Markov-chain or frequency approach:
```
python mc.py runs/test_mc.yaml
```

#### bo.py
Optimize uncertainty set size using Bayesian Optimization:
```
python bo.py runs/test_bo.yaml prefix_for_file_names
```

## References
Kondili, E.; Pantelides, C.; Sargent, R. A general algorithm for short-term scheduling of batch operations - I. MILP formulation. Computers & Chemical Engineering 1993, 17, 211227.

Wiebe, J.; Cecilio, I.; Misener, R. Robust optimization of processes with
degrading equipment. 2018 (Submitted).
