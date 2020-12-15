import numpy as np
import gym, sys, os
from nn import neuralNetwork
import random

env = gym.make("CartPole-v1")
Qmodel = neuralNetwork(env.observation_space.shape[0], 15, env.action_space.n, 0.3)
# Tmodel = neuralNetwork()
state = env.reset()
print(Qmodel.query(state).T[0])

EPISODES = 500
ALPHA = 0.6
GAMMA = 0.99
EPSILON = 0.6

MEMORY = []
BATCH_SIZE = 256

def main():
    for i in range(10):
        state = env.reset()
        done = False
        rew = 0
        while not done:
            env.render()
            next_state, reward, done, _ = env.step(np.argmax(Qmodel.query(state).T[0]))
            state = next_state
            rew += reward
        print(f"Testing Episode {i} has {rew} rewards")

    env.close()

def remember(s, a, r, ns, d):
    MEMORY.append([s, a, r, ns, d])

def experience_replay():
    global EPSILON
    if len(MEMORY) < BATCH_SIZE:
        return

    batch = random.sample(MEMORY, BATCH_SIZE)
    for state, action, reward, next_state, done in batch:
        q_update = reward
        if not done:
            q_update = (reward + GAMMA * np.max(Qmodel.query(next_state).T[0]))
        q_values = Qmodel.query(state).T[0]
        q_values[action] = q_update
        Qmodel.train(state, q_values)
        # if EPSILON > 0.1:
        #     EPSILON -= 0.001


def cartpole():
    for i in range(EPISODES):
        state = env.reset()
        done = False # not in tuto
        rew = 0
        while True:
            # env.render()
            action = env.action_space.sample() if np.random.random() < EPSILON else np.argmax(Qmodel.query(state).T[0])

            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -reward
            rew += reward
            remember(state, action, reward, next_state, done)
            experience_replay()
            state = next_state
            if done:
                break
        print(f"Episode {i} with reward {rew} episilon {EPSILON}")

print(Qmodel.query(env.reset()))

if __name__ == '__main__' and input("Press ENTER To Start\n") == "":
    try:
        main()
        cartpole()
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
