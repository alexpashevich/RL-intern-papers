# RL-intern-papers

This file consists of four sections where (1) grasping-realted, (2) RL in robotics, (3) continuous high-dimensional action space RL and (4) general deep RL papers correspondingly are provided. Each section consists of summaries of several papers on the topic. This file aims to describe state-of-the-art methods that can be potentially useful during the internship.

## Table of Contents

 - [Grasping-related papers section](#grasping-related-papers-section)
 - [State-of-the-art Deep RL section](#state-of-the-art-deep-rl-section)
 - [Continuous high-dimensional action space RL](#continuous-high-dimensional-action-space-rl)
 - [RL in robotics](#rl-in-robotics)


## Grasping-related papers section

## Continuous high-dimensional action space RL

## RL in robotics

## State-of-the-art Deep RL section
### Trust Region Policy Optimization \cite{DBLP:journals/corr/SchulmanLMJA15}
The authors developed an algorithm for policy optimisation with guaranteed monotonic improvement. In practice, the optimization is constrained with KL divergence between two policies which guarantees to make small steps improving the original policy (theoretical results are provided). This helps us to avoid overfitting to the most recent batch of data. In the experiment sections author show that the algorithm works well for learning robotic control policies in simulated environment.

### High-Dimensional Continuous Control Using Generalized Advantage Estimation [Paper](https://arxiv.org/abs/1502.05477)
The authors propose to use an Actor-Critic algorithm for applications in simulated robotic locomotion with high-dimensional continuous action space. The critic is so called Generalized Advantage Estimator (GAE) which tells how much better the particular action in the particular state than the average behaviour. To estimate GAE the conjugate gradient algorithm is used. The actor (policy) is estimated via the Trust Region Policy Optimization described in the previous paper \cite{DBLP:journals/corr/SchulmanLMJA15}. The authors provide results of experiments on simulated robotic locomotion tasks which achieves impressive results.

### Continuous control with deep reinforcement learning \cite{DBLP:journals/corr/LillicrapHPHETS15}
The authors present an actor-critic algorithm, model-free, off-policy algorithm based on the deterministic policy gradient algorithm (DPG) \cite{icml2014c1_silver14} that operate over continuous action space. The critic (value function) and the actor (policy) are represented with neural networks. The method used to train the critic is similar to the DQN from the classical Atari games paper \cite{DBLP:journals/corr/MnihKSGAWR13}. The actor is trained using the DPG (a gradient descent algorithm). The authors show successful applications of the method to a range of problems including legged locomotion and car driving.

### Continuous Deep Q-Learning with Model-based Acceleration \cite{DBLP:journals/corr/GuLSL16}
The first contribution of the paper is the Normalized Advantage Function (NAF) algorithm where to deal with the continuous action space, the advantage fuction is represented as a quadratic function of the state. This allows to analytically find its maximum and deduce the policy. The second contribution is the model-based method similar to Dyna-Q approach when in addition to real actions, imaginary rollouts are made. The authors fit a linear model locally around the latest sets of example and show that this helps to accelerate the training on the early stages. The method is shown to work slightly better than the previous one, DPPG, \cite{DBLP:journals/corr/LillicrapHPHETS15} on majority of tasks such as three-joint reacher, peg insertion and locomotion. The authors claims that "NAF outperformed DDPG on particularly manipulation tasks that require precision and suffer less from the lack of multimodal Q-functions".

### Deep Reinforcement Learning for Robotic Manipulation \cite{Gu2016}
In the paper the previous approach \cite{DBLP:journals/corr/GuLSL16} is applied to a system with multiple robots where multiple workers collect training data and send to a central server. The training is done with the NAF method on simulated tasks and real-world random target reaching and door opening. The authors show that NAF and DDPG have roughly the same performance.

### Learning Continuous Control Policies by Stochastic Value Gradients \cite{NIPS2015_5796}
A novel model-based off-policy method for continuous action spaces is presented in the paper. The authors present a family of methods however the most promising is a model-based actor-critic approach. Important difference from the methods like DPG \cite{icml2014c1_silver14} is the fact that a stochastic policy is used which implies usage of the Jacobian of the reward function. Method is evaluated on simulated  physic-based tasks like Reacher, Gripper, Monoped, Half-Cheetah and Walker. The authors show that the method outperforms DPG \cite{icml2014c1_silver14} but do not compare it with more advanced methods.