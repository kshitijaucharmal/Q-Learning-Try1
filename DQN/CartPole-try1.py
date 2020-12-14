import numpy as np
import gym
from nn import neuralNetwork

env = gym.make("CartPole-v0")
model = neuralNetwork(3, 10, 1, 0.3)

def main():
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

if __name__ == "__main__" and input("Press ENTER To Start\n") == "":
    main()
