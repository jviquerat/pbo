# Policy-based optimization

PBO (policy-based optimization) is a degenerate policy gradient algorithm used for black-box optimization. It shares common traits with both DRL (deep reinforcement learning) policy gradient methods, and ES (evolution strategies) techniques. In this repository, we present a parallel PBO algorithm with covariance matrix adaptation, with applications to (i) the minimization of simple analytical functions, and (ii) the optimization of parametric control laws for the chaotic Lorenz attractor. The related pre-print can be found <a href="">here</a>. This paper formalizes the approach used in previous related works:

- Direct shape optimization through deep reinforcement learning (<a href="https://www.sciencedirect.com/science/article/pii/S0021999120308548">paper</a>, <a href="https://arxiv.org/pdf/1908.09885.pdf">pre-print</a> and <a href="https://github.com/jviquerat/drl_shape_optimization">github repository</a>),
- Single-step deep reinforcement learning for open-loop control of laminar and turbulent flows (<a href="https://arxiv.org/pdf/2006.02979.pdf">pre-print</a>),
- Deep reinforcement learning for the control of conjugate heat transfer with application to workpiece cooling (<a href="https://arxiv.org/pdf/2011.15035.pdf">pre-print</a>)

Below are some selected visuals of cases presented in the paper.

## Parabola function

We consider the minimization on a parabola defined in [-5,5]x[-5,5]. Below is the course of a single run, generation after generation, with a starting point in [2.5,2.5]:

<p align="center">
  <img width="900" alt="" src="https://user-images.githubusercontent.com/44053700/104418877-e9634b80-5577-11eb-8fa9-ca369d6c8059.gif">
</p>

## Rosenbrock function

The Rosenbrock function is here defined in [-2,2]x[-2,2]. It contains a very narrow valley, with a minimum in [1,1]. The shape of the valley makes it a hard optimization problem for many algorithms. Here is the course of a single run, generation after generation, with a starting point in [0.0,-1.0]:

<p align="center">
  <img width="900" alt="" src="https://user-images.githubusercontent.com/44053700/104337105-67cad980-54f5-11eb-9c43-2ff2bc2624d4.gif">
</p>

## Parametric control laws for the chaotic Lorenz attractor

We consider the equations of the Lorenz attractor with a velocity-based control term:

<p align="center">
  <img width="300" alt="" src="https://user-images.githubusercontent.com/44053700/113980948-5caeed80-9847-11eb-859a-14eda25e4322.jpg">
</p>

We make use of the following non-linear control with four free parameters:

<p align="center">
  <img width="300" alt="" src="https://user-images.githubusercontent.com/44053700/113981120-954ec700-9847-11eb-9c44-02ef90aacf74.jpg">
</p>

Two control cases are designed: the first one consists in forcing the system to stay in the x<0 quadrant, while the second one consists in maximizing the number of sign changes (cases inspired from <a href="https://research.tue.nl/files/146730787/Beintema_afstudeerverslag.pdf">this thesis</a>). Below is a comparison between the uncontrolled system (top), and the two controlled cases (bottom left, bottom right).

<p align="center">
  <img width="600" alt="" src="https://user-images.githubusercontent.com/44053700/113982301-e57a5900-9848-11eb-80cf-82149ff28595.gif">
</p>

<p align="center">
  <img width="450" alt="" src="https://user-images.githubusercontent.com/44053700/113981902-77359680-9848-11eb-892e-6555dc203d84.gif">
  <img width="450" alt="" src="https://user-images.githubusercontent.com/44053700/113981922-7dc40e00-9848-11eb-9d57-4dbf7995be5e.gif">
</p>
