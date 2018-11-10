[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Training

For this project, we have first version that contains a single agent of the Unity environment:


#### Solve Status

The task is episodic, and in order to solve the environment,   agent must get an average score of +30 over 100 consecutive episodes.
<h2>Installation</h2>

To set up your python environment to run the code in this repository, follow the instructions below.

<h3>For Installation Windows 64 bit with GPU </h3>
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

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!  



