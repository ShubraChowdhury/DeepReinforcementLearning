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

# Report - DDPG: Continuous Control

### Implementation Details

There are 3 main files ddpg_agent.py and model.py, and  Continuous-Control.ipynb. 

1. model.py: Architecture and logic for the neural networks implementing the actor and critic for the chosen DDPG algorithm.
    Actor model has 2 fully connected layer (of 400 and 300 units) and Critic has 3 fully connected layer (of 400, 300 and 100 units). In both case a 1D batch normalization has been used.Input and output layers sizes are determined by the state and action space.

Actor Model/Network architecture | Value
--- | ---
fc1_units | 400  
fc2_units | 300 

Critic Model/Network architecture | Value
--- | ---
fc1_units | 400  
fc2_units | 300 
fc2_units | 100
    
2. ddpg_agent.py: This program implements Agent class and OUNoise, Agent includes step() which saves experience in replay memory and use random sample from buffer to learn, act() which returns actions for given state as per current policy, learn() which Update policy and value parameters using given batch of experience tuples  which is used  to train the agent, and uses 'model.py' to generate the local and target networks for the actor and critic.

3. Continuous-Control.ipynb: Contains instructions for how to use the Unity ML-Agents environment,the environments contain brains which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1. The main training loop creates an agent and trains it using the DDPG (details below) until satisfactory results. 

	Saving Model: Notebook saves model at 3 stages 
	1. Saves for all episode checkpoint_actor_all.pth and checkpoint_critic_all.pth
	2. Saved after every 100 episode checkpoint_actor.pth and checkpoint_critic.pth
	3. Saved after score of 30 plus and number of episode equal or greater than 100 checkpoint_actor_plus_30.pth and checkpoint_critic_plus_30.pth


### Learning Algorithm

The agent is trained using the DDPG algorithm which is an off-policy algorithm. It is not possible to straightforwardly apply Q-learning to continuous action spaces, because in continuous spaces finding the greedy policy requires an optimization of at at every timestep; this optimization is too slow to be practical with large, unconstrained function approximators and nontrivial action spaces. Instead DDPG is an actor-critic approach based on the DPG algorithm (Silveret al., 2014).
The DPG algorithm maintains a parameterized actor function {mu(s|theta to power mu)} which specifies the current policy by deterministically mapping states to a specific action. The critic Q(s; a) is learned using the Bellman equation as in Q-learning. The actor is updated by following the applying the chain rule to the expected return from the start distribution J with respect to the actor parameters.

As with Q learning, introducing non-linear function approximators means that convergence is no longer guaranteed. However, such approximators appear essential in order to learn and generalize on large state spaces. NFQCA (Hafner & Riedmiller, 2011), which uses the same update rules as DPG but with neural network function approximators, uses batch learning for stability, which is intractable for large networks. A minibatch version of NFQCA which does not reset the policy at each update, as would be required to scale to large networks, is equivalent to the original DPG. DDGP is modifications to DPG, inspired by the success of DQN, which allow it to use neural network function approximators to learn in large state and action spaces online.

One challenge when using neural networks for reinforcement learning is that most optimization algorithms assume that the samples are independently and identically distributed. When the samples are generated from exploring sequentially in an environment this assumption no longer holds. Additionally, to make efficient use of hardware optimizations, it is essential to learn in minibatches, rather than online.

As in DQN, DDPG uses replay buffer to address these issues. The replay buffer is a finite sized cache R. Transitions were sampled from the environment according to the exploration policy and the tuple (st; at; rt; st+1) was stored in the replay buffer. When the replay buffer was full the oldest samples were discarded. At each timestep the actor and critic are updated by sampling a minibatch uniformly
from the buffer. Because DDPG is an off-policy algorithm, the replay buffer can be large, allowing the algorithm to benefit from learning across a set of uncorrelated transitions.

A major challenge of learning in continuous action spaces is exploration. An advantage of off policies  algorithms such as DDPG is that it can treat the problem of exploration independently from the learning algorithm.

###  Sudo Code Explanation 
![Algo][image1]


###   Training using Deep Deterministic Policy Gradient (DDPG)          
		 def ddpg(n_episodes=1000, max_t=1000, print_every=100,window_size=100):
		    scores_deque = deque(maxlen=window_size) # last 100 scores
		    episode_score_list = []                  # list containing scores from each episode
		    Overall_average = []                     #List of 100/window mean scores
		    scores = np.zeros(num_agents)            # initialize the score (for each agent)
		    start_time = time.time()

		    for i_episode in range(1, n_episodes+1):
			env_info = env.reset(train_mode=True)[brain_name]     # reset the environment 
			states = env_info.vector_observations                  # get the current state (for each agent)
			scores = np.zeros(num_agents)                          # initialize the score (for each agent)
			agent.reset()

			average_score = 0
			time_step = time.time()
			for ts in range(max_t):
			    actions = agent.act(states, add_noise=True)        # select an action (for each agent)
			    env_info = env.step(actions)[brain_name]           # send action to  environment
			    next_states = env_info.vector_observations         # get next state (for each agent)
			    rewards = env_info.rewards                         # get reward (for each agent)
			    dones = env_info.local_done                        # see if episode finished
			    scores += env_info.rewards                         # update the score (for each agent)
			    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
				agent.step(state, action, reward, next_state, done,ts)
			    states = next_states                               # roll over states to next time step
			    if np.any(dones):                                  # exit loop if episode finished
				break
			episode_score_list.append(scores) # List of all scores in Episodde for agents
			scores_deque.append(scores)    # Making a of  window of 100 scores
			average_score = np.mean(scores_deque)  # Average/mean of  100/window 
			Overall_average.append(average_score)  # List of 100/window mean scores
			print('\rEpisode {}, Average Score:{:.2f} Window Size:({:d}),Epi Score:{:.2f}, Max Score: {:.2f}, Min Score: {:.2f},Time per Episode: {:.2f}'\
			      .format(i_episode, average_score,len(scores_deque),np.max(scores), np.max(scores_deque),np.min(scores_deque), time.time() - time_step), end="\n")
			torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_all.pth')
			torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_all.pth')

			if i_episode % print_every == 0:
			    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
			    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
			    print('\rEpisode {}\tAverage Window Score for a window of 100: {:.2f}'.format(i_episode, average_score))   

			if average_score >= 30.0 and i_episode >= 100 :  
				end_time = time.time()
				torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_plus_30.pth')
				torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_plus_30.pth')
				print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}, training time: {}'\
				      .format(i_episode, average_score, end_time-start_time))
				break

		    return episode_score_list, Overall_average
    
### Hyperparameters:

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




### Training Output With Average Scores


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

### Training Reward/Score Plot
![Plot][image2]

###  Ideas for future work

I will try  test and implement PPO or D4PG  on Udacity's [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) environment to minimize dependencies on hyperparameter and noise,  


### References:
1. [DDPG Paper](https://arxiv.org/pdf/1509.02971.pdf)

2. [DDPG-pendulum implementation](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)

3. Udacity code guidance for agent and model (ddpg_agent.py and model.py)

4. Reinforcement Learning Book by Richard S. Sutton  and Andrew G. Barto

5. [Silver Lever Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller ](http://proceedings.mlr.press/v32/silver14.pdf)
