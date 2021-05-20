# Reinforcement Learning

## Introduction

Let there be a model \(agent/actor\) within an environment where the model could perform different actions according to different states. For each action made in a certain state, the environment would return a reward to the model. The goal is to let the model learn an optimal strategy to maximize future rewards \(i.e. perform a certain task\). 

## Notations

| Symbol | Meaning |
| :--- | :--- |
| $$s \in \mathcal{S}$$ | States |
| $$a \in \mathcal{A}$$ | Actions |
| $$r \in \mathcal{R}$$ | Reward \(a.k.a. feedback\) |
| $$P$$ | Transition probability between states |
| $$\pi(s)$$ | Policy, guideline for learning the optimal strategy, what action to take in a particular state |
| $$V(s)$$ | Value function used for predicting expected amount of future rewards for a particular \(state, action\) pair. |
| $$G_t$$ | The future reward \(a.k.a. return\), the discounted rewards after $$t$$ time-steps.  |



