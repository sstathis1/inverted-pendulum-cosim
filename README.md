# Inverted-Pendulum-CoSim
Implementation of a co-simulation master algorithm for the optimal control (LQR-infinite time horizon) of the dynamics of an inverted pendulum. 

This project was done as part of my thesis at Aristotle University of Thessaloniki at the Laboratory of Machine Dynamics.

## Master Script :
* **master.py** : Contains the master object for the explicit co-simulation of two models. It supports Jacobi / Gauss-Seidel communication schemes, error estimation using Richardson Extrapolation and error control using adaptive step size. Also supports multithreading whith Jacobi scheme.

## Model Scripts :
* **single_pendulum.py** : Contains the model of the non-linear dynamical system of the single pendulum on cart.
* **single_pendulum_controller.py** : Contains the linear equivalent of the non-linear system, the lqr gain equations as well as three distinct estimation techniques namely : predictive estimation, current estimation and kalman filter.

* **double_pendulum.py** : Contains the model of the non-linear dynamical system of the double pendulum on cart.
* **double_pendulum_controller.py** : Contains the linear equivalent of the non-linear system, the lqr gain equations as well as three distinct estimation techniques namely : predictive estimation, current estimation and kalman filter.

## Test Scripts :
* **single_pendulum_cosim.py** : Contains a test script that initializes a single pendulum and a controller object and then cosimulates them using explicit methods. Finally the results are ploted and an animaton is created of the non-linear system response.
* **double_pendulum_cosim.py** : Contains a test script that initializes a double pendulum and a controller object and then cosimulates them using explicit methods. Finally the results are ploted and an animaton is created of the non-linear system response.

## Results of a single pendulum : 
![single_pendulum](https://user-images.githubusercontent.com/96697827/174833504-05c0d42b-8edb-4b11-ab15-c61523934d5f.gif)

## Results of a double pendulum:
![double_pendulum](https://user-images.githubusercontent.com/96697827/174835030-94e91c6b-b215-43c2-a51f-aed51b0ad99c.gif)
