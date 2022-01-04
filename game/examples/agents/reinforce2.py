from model2 import QNetwork
from collections import deque
from dqn_agent import Agent
import numpy as np
import torch
import gym
from matplotlib import pyplot as plt
from datetime import datetime

class Reinforce2:

    env = None
    agent = Agent(state_size=8, action_size=4, seed=3)
    
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.env.seed(0)
        print('State shape: ', self.env.observation_space.shape)
        print('Number of actions: ', self.env.action_space.n)

    def dqn(self, n_episodes=5000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            state = self.env.reset()
            score = 0
            for t in range(max_t):
                action = self.agent.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=500.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(self.agent.network.state_dict(), 'models/network2.pth')
                #torch.save(self.agent, 'dqn_network.pth')
                break
        return scores


    def train(self):
        scores = self.dqn()

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

if __name__=="__main__":
    start = datetime.now()
    r2 = Reinforce2()
    r2.train()
    print("Done.  Elapsed time: {}".format(datetime.now()-start))