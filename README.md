# pbo

PBO (policy-based optimization) is a degenerate policy gradient algorithm used for black-box optimization. It shares common traits with both DRL (deep reinforcement learning) policy gradient methods, and ES (evolution strategies) techniques. In this repository, we present a parallel PBO algorithm with possible covariance matrix adaptation. This feature is directly adapted from the CMA-ES algorithm, using ideas from https://arxiv.org/abs/1810.02541 (although the point of view on how to adapt the method differs a bit).

## Method

To complete

Below, you can find optimization examples on simple analytical functions. All cases are averaged over 20 runs.

## Parabola

We first consider the minimization on a parabola defined in [-5,5]x[-5,5]. Here is the course of a single run, generation after generation:

<p align="center">
  <img width="900" alt="" src="https://user-images.githubusercontent.com/44053700/85527883-917ceb80-b60b-11ea-92a6-a75155a07135.gif">
</p>

The PBO algorithm can use either standard es method (```es```), diagonal (```cma-diag```) or full (```cma-full```) covariance matrix adaptation. Here, although the given function is isotropic, covariance matrix adaptation shows a great improvement over standard ES. The results of all three methods are given below (left plot), along with a test using ```cma-diag``` various number of cpus (right plot):

<p align="center">
  <img width="450" alt="" src="https://user-images.githubusercontent.com/44053700/85555656-6b187980-b626-11ea-98f5-83374647aeb2.png">
  <img width="450" alt="" src="https://user-images.githubusercontent.com/44053700/85729231-e9455080-b6f8-11ea-9320-7022f932bc0d.png">
</p>

## Rosenbrock function

The Rosenbrock function is usually defined in [-2,2]x[-1,3]. It contains a very narrow valley, with a minimum set in [1,1]: 

<p align="center">
  <img width="450" alt="" src="https://user-images.githubusercontent.com/44053700/85549682-84b6c280-b620-11ea-9da6-84ef9752121b.png">
</p>

Given the shape of the valley, ```es``` and ```cma-diag``` versions cannot efficiently reach the minimum. Below is a comparison of the three methods: 

<p align="center">
  <img width="450" alt="" src="https://user-images.githubusercontent.com/44053700/85560877-61dddb80-b62b-11ea-9d51-f82ab75cd853.png">
</p>

As can be seen, the flexibility of the full covariance matrix adaptation yields a successfull optimization, where ```es``` and ```cma-diag``` fail:

<p align="center">
  <img width="900" alt="" src="https://user-images.githubusercontent.com/44053700/85410633-a0aa5d80-b567-11ea-9383-42906cb0cab6.gif">
</p>
