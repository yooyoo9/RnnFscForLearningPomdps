# Using RNNs for the internal state representation when learning POMDPs

Semester Project at ETH Zurich, [paper link](https://github.com/yooyoo9/RnnFscForLearningPomdps/blob/main/RnnFscPaper.pdf)

### Abstract:
Partially Observable Markov Decision Process (POMDP) is the standard model that captures the partial-information structure in Reinforcement Learning, which is often found in settings where agents learn to act even though the complete state of the underlying system is not given.
It is well known that learning POMDPs is difficult in theory as it is both statistically and computationally intractable.
There has been some attempts in extending the theory of Markov Decision Processes (MDPs) into the partially observed setting.
However, most results rely on very restrictive assumptions about the environment. 
In this work, we investigate ways to improve upon the known algorithms for learning POMDPs by the use of Recurrent Neural Networks (RNNs) for the internal state representation.
We theoretically quantify the performance gap between RNNs and Finite State Controllers (FSCs) in certain hand-crafted environments. 
Empirically, we compare the two algorithms in different modified MDP environments.

### Project Sturcture:
```
.
├── RnnFscPaper.pdf
├── pomdp <main implementation>
│   ├── agents
│   │   ├── ac <variants of the AC algorithm>
│   │   │   ├── ...
│   │   └── td3 <variants of TD3>
│   │       ├── ...
│   │
│   ├── envs <POMDP wrappers>
│   │   ├── ...
│   │
│   ├── utils
│   │   ├── ...
│   │
│   ├── create_plots.py
│   ├── main.py <main script for experiments on TD3>
│   └── main_ac.py <main script for experiments on AC>
│
└── results <experimental results>
    ├── ...
```

### How to Run Experiments:
Set up environment for the first time:
```
pip install -r requirements.txt
cd pomdp
pip install -e .
```

To run experiments, go to the root directory and type (the default parameter can be used for result reproduction):
```
python pomdp/main.py
python pomdp/main_ac.py
```
The supported environments are: CartPole, HalfCheetah and Ant.
Results are saved in the ```experiments``` folder by default.