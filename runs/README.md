# Configuration files

Instances of the STN model can be specified using `.yaml` configuration files.
The folling parameters can be used:

| Parameter       | Description                                                                         | Scripts           |
| --------------- | ----------------------------------------------------------------------------------- | ----------------- |
| Ts              | Length scheduling horizon                                                           | all               |
| dTs             | Time step scheduling horizon                                                        | all               |
| stn             | Data file with [stn structure](../instances/README.md)                              | all               |
| rdir            | Path to result directory                                                            | all               |
| solverparams    | Dictionary of solver parameters                                                     | all               |
| max             | Dictionary with maximum value for each product demand                               | lhs.py            |
| min             | Dictionary with minimum value for each product demand                               | lhs.py            |
| N               | Number of LHS samples                                                               | lhs.py            |
| Tp              | Length planning period                                                              | rolling.py, mc.py |
| dTP             | Time step planning period                                                           | rolling.py, mc.py |
| TP              | Logistic regression data file                                                       | rolling.py, mc.py |
| prfx            | Prefix for result files                                                             | rolling.py, mc.py |
| [Product]       | List of demands for each planning period for [Product]                              | rolling.py, mc.py |
| alphas          | List of alpha values (uncertainty set size) for which to solve                      | rolling.py, mc.py |
| periods         | Dictionary: {rolling: # of rolling horizon periods, eval: # of periods to evaluate) | rolling.py        |
| periods         | # of periods to evaluate                                                            | mc.py             |
| freq            | True: Use frequency approach if eval > rolling, False: MC approach                  | rolling.py, mc.py |
| robust          | True: solve robust model, False: solve deterministic approximation                  | rolling.py, bo.py |
| ccm             | Cost of corrective maintenance for each unit                                        | bo.py             |
