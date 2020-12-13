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

env = gym.make("Taxi-v3")
env.reset()
for i in range(100):
    env.render()
    _, _, done, _ = env.step(env.action_space.sample())
    print(done)
    sleep(0.01)
    clear()
    if done:
        break

env.close()
