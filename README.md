# pbo

PBO (policy-based optimization) is a degenerate policy gradient algorithm used for black-box optimization. It shares common traits with both DRL (deep reinforcement learning) policy gradient methods, and ES (evolution strategies) techniques. In this repository, we present a parallel PBO algorithm with covariance matrix adaptation. This feature is directly adapted from the CMA-ES algorithm, using ideas from https://arxiv.org/abs/1810.02541 (although the point of view on how to adapt the method differs). The performance level of the method as presented here is comparable with that of CMA-ES.

## Parabola function

We first consider the minimization on a parabola defined in [-5,5]x[-5,5]. Here is the course of a single run, generation after generation, with a starting point in [2.5,2.5]:

<p align="center">
  <img width="900" alt="" src="https://user-images.githubusercontent.com/44053700/104418877-e9634b80-5577-11eb-8fa9-ca369d6c8059.gif">
</p>

## Rosenbrock function

The Rosenbrock function is here defined in [-2,2]x[-2,2]. It contains a very narrow valley, with a minimum in [1,1]. The shape of the valley makes it a hard optimization problem for many algorithms. Here is the course of a single run, generation after generation, with a starting point in [0.0,-1.0]:

<p align="center">
  <img width="900" alt="" src="https://user-images.githubusercontent.com/44053700/104337105-67cad980-54f5-11eb-9c43-2ff2bc2624d4.gif">
</p>
