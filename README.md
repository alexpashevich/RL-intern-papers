# RL-intern-papers

This file consists of four sections where (1) grasping-realted, (2) RL in robotics, (3) continuous high-dimensional action space RL and (4) general deep RL papers correspondingly are provided. Each section consists of summaries of several papers on the topic. This file aims to describe state-of-the-art methods that can be potentially useful during the internship.


## Table of Contents

 - [Grasping-related papers](#grasping-related-papers-papers)
 - [Continuous high-dimensional action space RL papers](#continuous-high-dimensional-action-space-rl-papers)
 - [RL in robotics papers](#rl-in-robotics-papers)
 - [General Deep RL papers](#general-deep-rl-papers)


## Grasping-related papers

- **Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection (Sergey Levine et al., 2016)** [Paper](https://arxiv.org/abs/1603.02199)

The authors collected 800,000 grasp attempts to train a large CNN "to predict the probability that task-space motion of the gripper will result in successful grasps, using only monocular camera images". In addition to the predicting CNN, they used a servoing function that samples actions using the cross-entropy method, assesses them with the CNN and control the robot. The authors provide the dataset which was collected by 14 independent robots.

- **Supersizing Self-supervision: Learning to Grasp from 50K Tries and 700 Robot Hours (Pinto & Gupta, 2015)** [Paper](https://arxiv.org/abs/1509.06825)

The authors reformulate the grasping problem as a regression task with 18-dimensional likelihood vector (coarsely discretized 180-degree space). The collected dataset has 50K samples and is available online. During every trial, a random region of the image and a random angle are sampled for which a robot executes a trial. After the data collection, a CNN is trained to approximate the samples. The authors compare their approach with some methods such as SVM, kNN, min eigenvalue and eigenvalue limit and claim their method to overcome the baselines.

- **Data-Driven Grasp Synthesis - A Survey (Bohg et al., 2013)** [Paper](http://arxiv.org/abs/1309.2660)

The authors divide grasping methods into two main categories: analytic and data driven. Among the data driven methods proposed classification is grasping of known objects (1), of familiar objects (2) and of unknown objects (3). One of the paragraphs is dedicated to RL-based methods.

- **Reinforcement Learning in Robotics: A Survey (Kober et al., 2013)** [Paper](http://www.ias.tu-darmstadt.de/uploads/Publications/Kober_IJRR_2013.pdf)

The authors describe main successes and challenges of RL in robotics by July of 2013. The methods are classified into value function and policy search approaches. Most of the state-of-the-art methods are not presented in the paper as they appeared later.








## RL in robotics papers

- **Learning to Poke by Poking: Experiential Learning of Intuitive Physics (Agrawal et al., 2016)** [Paper](https://arxiv.org/abs/1606.07419)
The authors propose to learn the "intuitive" physics: understanding how actions effect objects with CNNs. The idea is to infer an action (a poke) given two photos of the same object before and after the action. This is done using a siamese CNN predicting the poke and a dataset with 50K pokes.

- **End-to-End Training of Deep Visuomotor Policies (Sergey Levine et al., 2015)** [Paper](https://arxiv.org/abs/1504.00702)
The authors show a method to learn torques command directly from raw visual input end-to-end. The method consists of two main components which are a supervised learning algorithm to learn a policy with a CNN and a trajectory-centric RL algorithm that provides the supervision for the first part. The method "requires the full state to be known during training, but not at test time". The approach is shown to be extremely sample-efficient and very effective in both simulated environments and real robot systems.

- **Learning to Push by Grasping: Using multiple tasks for effective learning (Pinto & Gupta, 2016)** [Paper](https://arxiv.org/abs/1609.09025)
The authors show that training a CNN can be done on different tasks such as grasping and pushing. "This paper attempts to break the myth of task-specific learning and shows that multi-task learning is not only effective but in fact improves the performance even when the total amount of data is the same".

- **Deep Visual Foresight for Planning Robot Motion (Finn & Levine, 2016)** [Paper](https://arxiv.org/abs/1610.00696)

Authors developed a model to predict how sequence of actions will affect a position of certain point of an object using LSTM-CNN. They used a task of moving an object to specified position ...


- **Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images (Watter et al., 2015)** [Paper](https://arxiv.org/abs/1506.07365)

smth








## Continuous high-dimensional action space RL papers
- **Trust Region Policy Optimization (Schulman et al., 2015)** [Paper](https://arxiv.org/abs/1502.05477)

The authors developed an algorithm for policy optimisation with guaranteed monotonic improvement. In practice, the optimization is constrained with KL divergence between two policies which guarantees to make small steps improving the original policy (theoretical results are provided). This helps us to avoid overfitting to the most recent batch of data. In the experiment sections author show that the algorithm works well for learning robotic control policies in simulated environment.

- **High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2015)** [Paper](https://arxiv.org/abs/1506.02438)

The authors propose to use an Actor-Critic algorithm for applications in simulated robotic locomotion with high-dimensional continuous action space. The critic is so called Generalized Advantage Estimator (GAE) which tells how much better the particular action in the particular state than the average behaviour. To estimate GAE the conjugate gradient algorithm is used. The actor (policy) is estimated via the Trust Region Policy Optimization described in [(Schulman et al., 2015)](https://arxiv.org/abs/1502.05477). The authors provide results of experiments on simulated robotic locomotion tasks which achieves impressive results.

- **Continuous control with deep reinforcement learning (Lillicrap et al., 2016)** [Paper](http://arxiv.org/abs/1509.02971)

The authors present an actor-critic algorithm, model-free, off-policy algorithm based on the deterministic policy gradient algorithm (DPG) [(Silver et al., 2014)](http://jmlr.org/proceedings/papers/v32/silver14.pdf) that operate over continuous action space. The critic (value function) and the actor (policy) are represented with neural networks. The method used to train the critic is similar to the DQN from the classical Atari games paper [(Mnih et al., 2013)](https://arxiv.org/abs/1312.5602). The actor is trained using the DPG (a gradient descent algorithm). The authors show successful applications of the method to a range of problems including legged locomotion and car driving.

- **Continuous Deep Q-Learning with Model-based Acceleration (Gu et al., 2016)** [Paper](http://arxiv.org/abs/1603.00748)

The first contribution of the paper is the Normalized Advantage Function (NAF) algorithm where to deal with the continuous action space, the advantage fuction is represented as a quadratic function of the state. This allows to analytically find its maximum and deduce the policy. The second contribution is the model-based method similar to Dyna-Q approach when in addition to real actions, imaginary rollouts are made. The authors fit a linear model locally around the latest sets of example and show that this helps to accelerate the training on the early stages. The method is shown to work slightly better than the previous one, DPPG [(Lillicrap et al., 2016)](http://arxiv.org/abs/1509.02971), on majority of tasks such as three-joint reacher, peg insertion and locomotion. The authors claims that "NAF outperformed DDPG on particularly manipulation tasks that require precision and suffer less from the lack of multimodal Q-functions".

- **Deep Reinforcement Learning for Robotic Manipulation (Gu et al., 2016)** [Paper](http://arxiv.org/abs/1610.00633)

In the paper the previous approach [(Gu et al., 2016)](http://arxiv.org/abs/1603.00748) is applied to a system with multiple robots where multiple workers collect training data and send to a central server. The training is done with the NAF method on simulated tasks and real-world random target reaching and door opening. The authors show that NAF and DDPG have roughly the same performance.

- **Learning Continuous Control Policies by Stochastic Value Gradients (Heess et al., 2015)** [Paper](http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients.pdf)

A novel model-based off-policy method for continuous action spaces is presented in the paper. The authors present a family of methods however the most promising is a model-based actor-critic approach. Important difference from the methods like DPG [(Silver et al., 2014)](http://jmlr.org/proceedings/papers/v32/silver14.pdf) is the fact that a stochastic policy is used which implies usage of the Jacobian of the reward function. Method is evaluated on simulated physic-based tasks like Reacher, Gripper, Monoped, Half-Cheetah and Walker. The authors show that the method outperforms DPG but do not compare it with more advanced methods.








## General Deep RL papers
- **Deep Reinforcement Learning with Double Q-learning (Hasselt et al., 2015)** [Paper](https://arxiv.org/abs/1509.06461)

The authors show that the well-known property of the Q-learning algorithm to overestimate some actions can harm performance. Such overestimation can be reduced by using Double DQN. "The idea of Double Q-learning is to reduce overestimation by decomposing the max operation in the target into action selection and action evaluation". In the particular Double DQN setting, the current network is used to find the max action and the target network can be used to estimate it. The authors provide experiments which prove to reduce the overestimation and improve performance.

- **Prioritized Experience Replay (Schaul et al., 2015)** [Paper](https://arxiv.org/abs/1511.05952)

The authors of the paper propose a technique called Prioritized Experience Replay to use the samples stored in memory more efficiently. They propose to rank all the samples using the TD error with either proportional or rank-based prioritizations. This sampling introduces new bias which they suggest to remove with importance-sampling which means multiplying the gradient value by a sample-dependent weight. The authors show that their approach improves performance of the state-of-the-art Double Q-learning method.

- **Asynchronous Methods for Deep Reinforcement Learning (Mnih et al., 2016)** [Paper](https://arxiv.org/abs/1602.01783)

In the paper the authors propose to use asynchronous methods of training of DRL algorithm such as SARSA, 1 step and n steps Q-Learning and Advantage Actor-Critic [(Schulman et al., 2015)](https://arxiv.org/abs/1506.02438). They propose to train it on multiple CPUs of single machine rather than on GPU in a fashion where each core has its own copy of environment. After making some replays and performing on-line training, they update the global model using Hogwild style updates (hoping for the best, the parameters are not locked and updated asynchronously). The authors show that such technique allows us to remove the experience replay and outperform DQN trained on GPU. Among the used RL approaches, they claim the Advantage Actor-Critic method to achieve the best results.

- **Reinforcement Learning with Unsupervised Auxiliary Tasks (Jaderberg et al., 2016)** [Paper](http://arxiv.org/abs/1611.05397)

The authors contribute with the idea to speed up the learning by maximising some others pseudo-reward functions except the main goal of maximising future expected reward. The method used in the paper is the A3C method from the previous section. They built a system solving several auxiliary tasks such as changing the pixels values and predicting immediate reward based on some experience. The system shares weight among networks for separate tasks and learn features better and faster. Such approach shows not only faster learning but also impressively increased performance on Atari games and the Labyrinth tasks.

- **Curiosity-driven Exploration in Deep Reinforcement Learning via Bayesian Neural Networks (Houthooft et al., 2016)** [Paper](https://arxiv.org/abs/1605.09674)

The authors propose to use a more sophisticated algorithm to explore rather than epsilon-greedy or adding some noise to actions. They propose to choose the action which provides the most information about the environment. The environment itself is modeled by a Bayesian Neural Network using the Variational Bayes approach.

- **Dueling Network Architectures for Deep Reinforcement Learning (Wang et al., 2015)** [Paper](https://arxiv.org/abs/1511.06581)

The authors argue that the traditional architecture of NN for evaluating a value function is not the optimal one and propose to separately estimate the value function and the advantage function by separating the stream of NN in two. Such approach outperforms baselines and was successfully used in [(Gu et al., 2016)](http://arxiv.org/abs/1603.00748).

- **Deep Exploration via Bootstrapped DQN (Osband et al., 2016)** [Paper](https://arxiv.org/abs/1602.04621)

Deep exploration is done using a shared architecture of DQN where K bootstrapped "heads" branching off independently. Thus, K approximations of the Q functions are made with a single NN which add randomness and improve the exploration.
