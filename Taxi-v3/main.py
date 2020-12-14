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

def main():
    EPISODES = 1000

    env = gym.make("Taxi-v3")

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    EPSILON = 0.6
    GAMMA = 0.99
    ALPHA = 0.6

    tot_rew = []
    for i in range(EPISODES):
        state = env.reset()
        done = False
        rew = 0
        while not done:
            action = env.action_space.sample() if np.random.random() < EPSILON else np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            Q[state, action] += ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            rew += reward
        print(f"Interation {i} with rewards {rew}")
        clear()
        tot_rew.append(rew)

    if(input("Press Enter to get info") == ''):
        print(f"Average Rewards from {EPISODES} episodes is {np.sum(tot_rew)/len(tot_rew)}")
        print("Q table : ")
        print(Q)

    if(input("Press Enter to continue or q to quit") == "q"):
        exit()

    tot_rew = []
    for i in range(10):
        state = env.reset()
        done = False
        rew = 0
        while not done:
            env.render()
            action = np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            rew += reward
            state = next_state
            sleep(0.1)
            clear()
        print(f"Iteration {i} with reward {rew}")
        tot_rew.append(rew)
    print("Total Reward after training {}".format(sum(tot_rew) / len(tot_rew)))

if __name__ == "__main__":
    main()
