###Honours Project ###

Model-free Reinforcement Learning (RL) has had an enormous impact in domains where reward
functions are well-defined and actions are discrete (see MuZero) [1]. This success has not been fully
transferred over to continuous domains such as robotics. We have seen teacher-student policies learn
and generalise diﬀerent behaviours for quadruped robots but this has been at the cost of very long and
ineﬃcient training runs. Optimisation-based control is model-based and uses derivative information,
allowing it to generate clean intricate behaviours [2]. However, the methods need to be more scalable,
without needing to solve problems every few milliseconds. Recent approaches have shown that we can
leverage derivative information and diﬀerentiable simulation to learn value/policy functions much more
eﬃciently than model-free RL [3, 4, 5]. These results however are limited and do not explore the
tradeoﬀs. In this project, we aim to leverage diﬀerentiable simulators [6, 7] and any gradient-based
methods to benchmark against widely used RL algorithms on complex nonlinear tasks.
