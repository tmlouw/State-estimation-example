# State Estimation Example: Leaching Reactor

A collection of Jupyter notebooks illustrating state estimation techniques using a nonlinear leaching reactor as a case study.

## Overview

This repository demonstrates how to estimate unmeasurable (latent) states from noisy, indirect measurements using classical filtering methods. The example system is a continuous-flow leach reactor where:
- Observable measurement: pH (measured with noise)
- Hidden states to estimate: dissolved copper concentration, acid concentration, and solid-phase copper concentration

## Leaching Model

The leaching process is modeled as a well-mixed continuous stirred-tank reactor (CSTR) with two inlet streams:

<div align="center">
  <img src="figures/PFD.png" width="250">
</div>

$$\frac{dC_{\text{Cu}}^{(s)}}{dt} = \frac{F}{V}(C_{\text{Cu},in}^{(s)} - C_{\text{Cu}}^{(s)}) - r$$

$$\frac{dC_{\text{Cu}}^{(aq)}}{dt} = -\frac{F}{V} C_{\text{Cu}}^{(aq)} + r$$

$$\frac{dC_{\text{H}}}{dt} = \frac{F}{V}(C_{\text{H},in} - C_{\text{H}}) - \gamma r + \frac{q_H}{V}$$

where the dissolution reaction rate is:

$$r = k \cdot C_{\text{H}} \cdot C_{\text{Cu}}^{(s)}$$

and pH is measured with additive Gaussian noise:

$$\text{pH} = -\log_{10}(C_{\text{H}}) + v, \quad v \sim \mathcal{N}(0, R)$$

Model parameters:
- $F$ = slurry volumetric flow rate (m³/min)
- $V$ = reactor volume (m³)
- $k$ = reaction rate constant (m³/(min·mol))
- $\gamma$ = stoichiometric acid consumption per mole Cu dissolved
- $C_{\text{H},in}$ = acid concentration in feed (mol/m³)
- $C_{\text{Cu},in}^{(s)}$ = solid copper concentration in feed (mol/m³)
- $q_H$ = acid dosing rate (mol/min)

## Notebooks

### 1. bayes-rule-illustrated.ipynb
An introduction to Bayesian inference and how prior knowledge is updated with noisy observations.

### 2. leaching_ekf.ipynb
Extended Kalman Filter (EKF) implementation for the leaching reactor.

- Method: A hybrid continuous-discrete EKF that integrates nonlinear dynamics and covariance evolution between measurement updates
- Key features:
  - Automatic differentiation (JAX) for computing Jacobians
  - Mean prediction via ODE integration: $\frac{d\hat{\mathbf{x}}}{dt} = f(\hat{\mathbf{x}}, \mathbf{u})$
  - Covariance propagation via the Riccati equation: $\frac{dP}{dt} = AP + PA^\top + Q$
  - Discrete Kalman update step using the measurement Jacobian $C = \partial h/\partial x$
  - Handles a step change in acid dosing to demonstrate disturbance tracking

### 3. leaching_pf_animation.ipynb
Particle Filter implementation with animated visualization.

- Method: Sequential importance resampling for nonlinear, non-Gaussian state estimation
- Shows how an ensemble of weighted particles approximates the posterior state distribution
- Animated plots of particle clouds tracking the true state over time

## Dependencies

- `numpy` — numerical computing
- `scipy` — ODE integration (`solve_ivp`)
- `jax` — automatic differentiation for computing Jacobians
- `matplotlib` — plotting
