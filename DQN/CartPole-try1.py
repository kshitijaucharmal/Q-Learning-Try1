import numpy as np
import gym

env = gym.make("CartPole-v0")

for i in range(10):
    env.reset()
    done = False
    rew = 0
    while not done:
        env.render()
        next_state, reward, done, _ = env.step(env.action_space.sample())
        rew += reward
    print(f"Episode {i} has {rew} rewards")

env.close()
