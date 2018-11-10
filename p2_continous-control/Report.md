[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/15965062/47237461-d2a90b00-d3e7-11e8-96a0-f0c9a0b7ad1d.png "Algorithm"
[image2]: https://user-images.githubusercontent.com/15965062/47245012-40613100-d400-11e8-904e-5b732c8c2871.png "Plot of Rewards"

# Report - Deep RL Project: Continuous Control

### Implementation Details

There are 3 main files 'ddpg_agent.py' and 'model.py', and  'Continuous-Control.ipynb'. 

1. 'model.py': Architecture and logic for the neural networks implementing the actor and critic for the chosen DDPG algorithm.
    Actor model has 2 fully connected layer and Critic has 3 fully connected layer. In both case a 1D batch normal has been used.

Actor Model | Value
--- | ---
fc1_units | 400  
fc2_units | 300 

Critic Model | Value
--- | ---
fc1_units | 400  
fc2_units | 300 
fc2_units | 100

2. 'ddpg_agent.py': Implements the agent class, which includes the logic for the stepping, acting, learning and the buffer to hold the experience data on which to train the agent, and uses 'model.py' to generate the local and target networks for the actor and critic.

3. 'Continuous-Control.ipynb': Main training logic and usage instructions. Includes explainations about the environment, state and action space, goals and final results. The main training loop creates an agent and trains it using the DDPG (details below) until satisfactory results. 

### Learning Algorithm

The agent is trained using the DDPG algorithm.

References:
1. [DDPG Paper](https://arxiv.org/pdf/1509.02971.pdf)

2. [DDPG-pendulum implementation](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)

3. Algorithm details: 

![Algorithm][image1]


4. Short explanation (refer to the papers for further details):
    - Q-Learning is not straighforwardly applied to continuous tasks due to the argmax operation over infinite actions in the continuous domain. DDPG can be viewed as an extension of Q-learning to continuous tasks.

    - DDPG was introduced as an actor-critic algorithm, although the roles of the actor and critic here are a bit different then the classic actor-critic algorithms. Here, the actor implements a current policy to deterministically map states to a specific "best" action. The critic implemets the Q function, and is trained using the same paradigm as in Q-learning, with the next action in the Bellman equation given from the actor's output. The actor is trained by the gradient from maximizing the estimated Q-value from the critic, when the actor's best predicted action is used as input to the critic.
    
    - As in Deep Q-learning, DDPG also implements a replay buffer to gather experiences from the agent (or the multiple parallel agents in the 2nd version of the stated environment). 
    
    - In order to encourage exploration during training, Ornstein-Uhlenbeck noise is added to the actors selected actions. I also needed to decay this noise using an epsilon hyperparameter to achieve best results.
    
    - Another fine detail is the use of soft updates (parameterized by tau below) to the target networks instead of hard updates as in the original DQN paper. 
    
5. Hyperparameters:

Parameters | Value
--- | ---
Replay buffer size | int(1e6)
Minibatch size | 512
Discount factor | 0.99  
Tau (soft update) | 1e-3
Learning rate actor | 1e-3
Learning rate critic | 1e-3
L2 weight decay | 0
Noise Sigma | 0.05
Theta | 0.15
Mu | 0

Training Parameters | Value
--- | ---
Number of episodes | 1000
Max_t | 1000
Print every |100
Deque Window |100 


6. Network architecture:
    - Both the actor and critic are implemented using fully connected networks, with 2 hidden layers of 128 units each, batch normalization and Relu activation function, with Tanh activation at the last layer.
    - Input and output layers sizes are determined by the state and action space.
    - Training time until solving the environment takes around 38 minutes on AWS p2 instance with Tesla k80 GPU.
    - See 'model.py' for more details.

### Plot of results

As seen below, the environment is solved after 129 episodes (average over agents over episodes 30-129 > 30.0), and achieves best average score of above 37.

Episodes | Average Score |Window Size|Max Episode Score| Max Deque| Min Deque| Time
--- | --- | --- | --- | ---| --- | --- 
--- | --- | --- | --- | ---| --- | ---
Episode 1| Average Score:0.54 |Window Size:(1)|Epi Score:0.54| Max Score: 0.54| Min Score: 0.54 |Time per Episode: 12.81
Episode 2| Average Score:0.30 |Window Size:(2)|Epi Score:0.07| Max Score: 0.54| Min Score: 0.07|Time per Episode: 19.17
Episode 3| Average Score:0.35 |Window Size:(3)|Epi Score:0.43| Max Score: 0.54| Min Score: 0.07|Time per Episode: 19.14
Episode 173| Average Score:18.10 |Window Size:(100)|Epi Score:31.95| Max Score: 37.94| Min Score: 4.87|Time per Episode: 20.30
Episode 174| Average Score:18.21 |Window Size:(100)|Epi Score:19.14| Max Score: 37.94| Min Score: 4.87|Time per Episode: 20.43
Episode 175| Average Score:18.52 |Window Size:(100)|Epi Score:35.92| Max Score: 37.94| Min Score: 6.00|Time per Episode: 20.54
Episode 176| Average Score:18.74 |Window Size:(100)|Epi Score:28.70| Max Score: 37.94| Min Score: 6.00|Time per Episode: 20.49
Episode 177| Average Score:18.92 |Window Size:(100)|Epi Score:29.70| Max Score: 37.94| Min Score: 6.00|Time per Episode: 20.45
Episode 178| Average Score:19.10 |Window Size:(100)|Epi Score:24.35| Max Score: 37.94| Min Score: 6.00|Time per Episode: 20.45
Episode 179| Average Score:19.28 |Window Size:(100)|Epi Score:23.77| Max Score: 37.94| Min Score: 6.37|Time per Episode: 20.52
Episode 180| Average Score:19.45 |Window Size:(100)|Epi Score:27.18| Max Score: 37.94| Min Score: 6.37|Time per Episode: 20.66
Episode 181| Average Score:19.71 |Window Size:(100)|Epi Score:34.58| Max Score: 37.94| Min Score: 6.37|Time per Episode: 20.72
Episode 244| Average Score:29.57 Window Size:(100)|Epi Score:31.44| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.38
Episode 245| Average Score:29.69 Window Size:(100)|Epi Score:33.36| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.44
Episode 246| Average Score:29.81 Window Size:(100)|Epi Score:36.15| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.56
Episode 247| Average Score:29.87 Window Size:(100)|Epi Score:34.11| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.49
Episode 248| Average Score:29.82 Window Size:(100)|Epi Score:25.55| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.52
Episode 249| Average Score:29.88 Window Size:(100)|Epi Score:36.02| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.47
Episode 250| Average Score:29.76 Window Size:(100)|Epi Score:13.99| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.59
Episode 251| Average Score:29.79 Window Size:(100)|Epi Score:32.33| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.62
Episode 252| Average Score:29.85 Window Size:(100)|Epi Score:29.69| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.63
Episode 253| Average Score:29.88 Window Size:(100)|Epi Score:30.85| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.51
Episode 254| Average Score:29.96 Window Size:(100)|Epi Score:34.43| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.49
Episode 255| Average Score:30.09 Window Size:(100)|Epi Score:27.95| Max Score: 38.43| Min Score: 6.68|Time per Episode: 21.55


Environment solved in 255 episodes!	Average Score: 30.09, total training time: 5173.3459 seconds


![Plot of Rewards][image2]

###  Ideas for future work

1. This DDPG implementation was very dependent on hyperparameter, noise settings and random seed. Solving the environment using PPO, TRPO or D4PG might allow a more robust solution to this task.

2. Solving the more challenging [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) environment using edited versions of these same algorithms. 

I'll at least do PPO and attempts to solve the Crawler environment after submission of this project (due to Udacity project submission rules).
