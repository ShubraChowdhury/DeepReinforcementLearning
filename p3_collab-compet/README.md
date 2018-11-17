[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

#### Solve Status
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

To set up your python environment to run the code in this repository, follow the instructions below.

<h3>For Installation Windows 64 bit with GPU (Windows 10 ) </h3>
<h4>1. Create (and activate) a new environment with Python 3.6.</h4>

conda create --name drlnd python=3.6 
  <br>
activate drlnd<br>
<h4>2. Install below mentioned python packages</h4>
 
matplotlib 2.2.2  <br>
numpy 1.14.3 <br>
scipy 1.1.0<br>
pandas  0.23.0<br>
jupyter   1.0.0 <br> 
ipykernel  4.8.2 <br> 
ipython   6.4.0  <br>
libboost    1.65.1 <br> 
llvmlite    0.23.1 <br> 
docopt  0.6.2<br>
cython   0.28.2<br>
tensorflow==1.7.1<br>
cuda90  1.0  h4c72538_0  peterjc123<br>
torch-0.4.0 (torch-0.4.0-cp36-cp36m-win_amd64.whl)<br>
torchvision    0.2.1<br>
 pytorch 0.3.1<br>
<h4>3. Follow the instructions in this <a href="https://github.com/openai/gym">repository</a> to perform a minimal install of OpenAI gym. </h4> 
 Box2D  2.3.2   <br>
Box2D-kengz  2.3.3<br>
atari-py 0.1.1 (atari_py-0.1.1-cp36-cp36m-win_amd64.whl )<br>
<h4>4. Clone the repository deep-reinforcement-learning.git</h4>
  git clone https://github.com/udacity/deep-reinforcement-learning.git<br>
cd deep-reinforcement-learning/python<br>
pip install .<br>

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  

### References:
1. [DDPG Paper](https://arxiv.org/pdf/1509.02971.pdf)

2. [DDPG-pendulum implementation](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)

3. Udacity code guidance for agent and model (ddpg_agent.py and model.py)

4. Reinforcement Learning Book by Richard S. Sutton  and Andrew G. Barto

5. [Silver Lever Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller ](http://proceedings.mlr.press/v32/silver14.pdf)
