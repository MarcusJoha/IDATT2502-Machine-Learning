# %%
"""
## Oblig 8, Reinforcement Learning
"""

# %%
import gym
import numpy as np
import time, math, random
from typing import Tuple

from sklearn.preprocessing import KBinsDiscretizer


env = gym.make('CartPole-v1')

env.reset()

# %%
# Crash python kernel for some reason
# Just to visualize the enviroment so no need to fix this
policy = lambda obs: 1 

for _ in range(5):
    env.reset()
    for _ in range(80):
        env.render()
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        # time.sleep(0.05)
        if done:
            env.reset()
            break
env.close()
exit()


    

# %%
# Enkel policy uten q learning
# Beveger bare klossen i retningen til stolpen 
policy = lambda _,__,___, tip_velocity : int(tip_velocity > 0)

# %%
"""
##### Q-learning Solution
"""

# %%
n_bins = (6, 12)
lower_bounds = [env.observation_space.low[2], -math.radians(50)]
upper_bounds = [env.observation_space.high[2], math.radians(50)]

def discretizer( _ , __ , angle, pole_velocity ) -> Tuple[int,...]:
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform') # continious space to discrete space
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))

# %%
# Initialise the Q values with zero
Q_table = np.zeros(n_bins + (env.action_space.n,))

# %%
# policy function, uses the Q-table to and greedly selecting the highest q-value
def policy(state : tuple):
    # Choosing action based on epsilor-greedy ploicy
    return np.argmax(Q_table[state])

# %%
# Update function
def new_Q_value(reward, new_state, discount_factor=1) -> float:
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor + future_optimal_value
    return learned_value

# %%
# Decaying learning rate
# Learn faster at the start and slower at the end
def learning_rate(n: int, min_rate=0.01 ) -> float:
    return max(min_rate, min(1,1.0 - math.log((n+1)/25)))

# %%
# Exploration rate
def exploration_rate(n: int, min_rate=0.1) -> float:
    return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))

# %%
"""
##### Learning
"""

# %%
n_episodes = 2000 
for e in range(n_episodes):
    
    # Siscretize state into buckets
    current_state, done = discretizer(*env.reset()), False
    
    while done==False:
        
        # policy action 
        action = policy(current_state) # exploit
        
        # insert random action
        if np.random.random() < exploration_rate(e) : 
            action = env.action_space.sample() # explore 
         
        # increment enviroment
        obs, reward, done, _ = env.step(action)
        new_state = discretizer(*obs)
        
        # Update Q-Table
        lr = learning_rate(e)
        learnt_value = new_Q_value(reward , new_state )
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1-lr)*old_value + lr*learnt_value
        
        current_state = new_state
        
        # Render the cartpole environment
        if (e %500 == 0):
            print("Round: {}, and done: {}, reward: {}".format(e, done, reward))
        
        if (e > 1700):
            env.render()

# %%
"""
#### Teori
* q-læring forsøker å lære en policy som makismerer total belønning.
* q = quality, altså hvor nyttig en handling er for å få en belønning
* Q-table -> [state, actions], hvor mange state i dette eksempelet? 180 grader?
* exploiting -> handling basert på informasjon tilgjengelig for oss
* exploring -> random handling for å utforske. (epsilon)

* Update q valuesQ[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])

##### Begreper
* Learningrate
    * alpha, terskelen for å akseptere en ny verdi sammenlignet med en gammel verdi. (new - old) * lr.
* Gamma
    * Discount factor, blansere immediate reward vs future reward.
* np.max()
    * maximum av fremtidig belønning and applaying it to the current state. Påvirker current action av mulige fremtidige belønninger.
"""