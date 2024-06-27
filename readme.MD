# Generalised Water Management System: Reinforcement Learning formulation and implementation as Gymnasium Environment

This Github repository contains the implementation described in "Bottom-Up Formulation of Water Management Systems as a Reinforcement Learning Problem", which can be found as pdf in this repository. Its goal is to provide a generalised formulation and implementation of Water Management Systems (WMSs) as a Reinforcement Learning (RL) problem. This way, WMS researchers no longer have to write simulations from scratch, as this implementation is applicable to a wide range of WMSs. The RL problem is implemented as a [Gymnasium Environment](https://gymnasium.farama.org/), specifically a [multi-objective Gymnasium Environment](https://mo-gymnasium.farama.org/).

The general definition of a WMS is based on three case studies:

- Nile River in Ethiopia, Sudan and Egypt by [Sari (2022)](https://repository.tudelft.nl/islandora/object/uuid%3Aba09fa13-5331-446c-bf5e-48f6a29042ca?collection=education)
- Lower Volta River in Ghana by [Owusu et al. (2023)](https://hess.copernicus.org/articles/27/2001/2023/)
- Susquehanna River in USA by [Witvliet (2022)](https://repository.tudelft.nl/islandora/object/uuid%3Abb16cada-c56f-4703-aad2-93c0ad9e75bb)

## Contents

### Basic WMS

The _/basic_ folder contains the most basic formulation of a WMS: water moving from A to B. It is represented by a graph structure, where nodes represent water points and edges represent flows.

### Generalised formulation and implementation of WMS as RL problem

The _/core_ folder contains the more advanced features of the proposed implementation:

- _graph.py_ contains the `Node` and `Edge` classes which implement the most complex and advanced water management mechanics. They are at the building blocks of the RL environment
- _reward.py_ contains the `Reward` data type which can be used to create custom reward functions.
- _watermanagementsystem.py_ contains the implementation of Gymnasium's `Env` class.
- _utils.py_ provides some useful functions for, for instance, finding current time-dependent values or generating a `release_range_function`
- _main.py_ provides an example of a WMS implementation with 4 nodes.

### Comparison with Nile River simulation

As described in the paper, the implementation's accuracy is compared to that of the studied [Nile River simulation](https://github.com/ysari97/eastern-nile-optimization/tree/main) from [Yasin Sari (2022)](https://repository.tudelft.nl/islandora/object/uuid%3Aba09fa13-5331-446c-bf5e-48f6a29042ca?collection=education). To do so, the same data, which can be found in _/data_, and actions are used. The simulation is written in _nile_main.py_, using some utils functions from _nile_utils.py_ and is run in _nile_verify.py_. The results are stored in a .csv file and compared in _comparison.ipynb_, which can all be found in the _/verification_ folder.

## Questions & comments

For any questions or comments, feel free to contact the author using the below contact details. I would be more than happy to help!

Contact details: jorian.faber@gmail.com
