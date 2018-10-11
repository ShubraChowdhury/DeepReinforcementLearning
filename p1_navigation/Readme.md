<h1>The Environment</h1>
For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

![image](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif )




A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

<h2>Installation</h2>

To set up your python environment to run the code in this repository, follow the instructions below.

<h3>For Installation Windows with GPU </h3>
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
<h4>4. Clone the repository deep-reinforcement-learning.git</h4><br>
  git clone https://github.com/udacity/deep-reinforcement-learning.git<br>
cd deep-reinforcement-learning/python<br>
pip install .<br>
