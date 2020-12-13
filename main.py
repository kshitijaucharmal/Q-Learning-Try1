import gym
import numpy as np
from time import sleep
from os import system, name

def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

EPISODES = 4

env = gym.make("Taxi-v3")
tot_rew = []
for i in range(EPISODES):
    env.reset()
    done = False
    rew = 0
    while not done:
        env.render()
        next_state, reward, done, _ = env.step(env.action_space.sample())
        print(f"Iteration {i} Done {done}")
        rew += reward
        sleep(0.02)
        clear()
    tot_rew.append(rew)

if(input("Press Enter to get info and exit") == ''):
    print(f"Average Rewards from {EPISODES} episodes is {np.sum(tot_rew)/len(tot_rew)}")
    env.close()
