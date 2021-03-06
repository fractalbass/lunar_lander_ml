import argparse
import gym
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions import Categorical
from itertools import count
from datetime import datetime

# if gpu is to be used
use_cuda = torch.cuda.is_available()
gpu_id = 0
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Policy(nn.Module):
    def __init__(self, state_size, num_actions):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x),dim=0)
        return x

class Reinforce(object):
    def __init__(self, env, args):
        super(Reinforce, self).__init__()
        self.env = env
        self.policy = Policy(env.observation_space.shape[0], env.action_space.n)
        if use_cuda:
            self.policy.cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.num_episodes = args.num_episodes
        self.test_episodes = args.test_episodes
        self.num_steps = args.num_steps
        self.gamma = args.gamma
        self.expt_name = args.expt_name
        self.save_path = args.save_path
        self.test_freq = args.test_freq
        self.save_freq = args.save_freq
        self.train_rewards = []
        self.test_rewards = []
        self.train_steps = []
        self.test_steps = []
        self.losses = []

    def select_action(self, state):
        state = Variable(Tensor(state))
        action_probs = self.policy(state)
        log_probs = action_probs.log()
        action = Categorical(action_probs).sample()
        return action.data.cpu().numpy(), log_probs[action]

    def play_episode(self, e):
        state = self.env.reset()
        steps = 0
        rewards = []
        log_probs = []
        while True:
            action, log_prob = self.select_action(state)
            state, reward, is_terminal, _ = self.env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            steps +=1
            if is_terminal:
                break
        return steps, rewards, log_probs

    def optimize(self, rewards, log_probs):
        R = torch.zeros(1,1).type(FloatTensor)
        loss = 0
        for i in reversed(range(len(rewards))):
            # downscaling rewards by 1e-2 to help training
            R = self.gamma * R + (rewards[i] * 1e-2)
            loss = loss - (log_probs[i]*Variable(R))
        loss = loss / len(rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.detach().cpu().numpy())

    def train(self, num_episodes):
        print("Going to be training for a total of {} episodes".format(num_episodes))
        state = Variable(torch.Tensor(self.env.reset()))
        for d in range(torch.cuda.device_count()):
            print("Cuda device {}: {}".format(d, torch.cuda.get_device_name(d)))

        torch.cuda.set_device(gpu_id)
        print("Using device: {}".format(gpu_id))

        for e in range(num_episodes):
            start = datetime.now()
            steps, rewards, log_probs = self.play_episode(e)
            self.train_rewards.append(sum(rewards))
            self.train_steps.append(steps)
            self.optimize(rewards, log_probs)

            if (e+1) % 100 == 0:
                print("Episode: {}, reward: {}, steps: {}".format(e+1, sum(rewards), steps))
                print("Training Time/Episode: {}".format((datetime.now()-start)/100))
                start=datetime.now()

            # Freeze the current policy and test over 100 episodes
            if (e+1) % self.test_freq == 0:
                print("-"*10 + " testing now " + "-"*10)
                self.test(self.test_episodes)

            # Save the current policy model
            if (e+1) % (self.save_freq) == 0:
                torch.save(self.policy.state_dict(), os.path.join(self.save_path, "train_ep_{}.pkl".format(e+1)))
            
        # plot once when done training
        self.plot_rewards(save=True)

    def test(self, num_episodes):
        state = Variable(torch.Tensor(self.env.reset()))
        testing_rewards = []
        testing_steps = []
        for e in range(num_episodes):
            steps, rewards, log_probs = self.play_episode(e)
            self.test_rewards.append(sum(rewards))
            self.test_steps.append(steps)
            testing_rewards.append(sum(rewards))
            testing_steps.append(steps)
        print("Mean reward achieved : {} ".format(np.mean(testing_rewards)))
        print("-"*50)
        if np.mean(testing_rewards) >= 200:
            print("-"*10 + " Solved! " + "-"*10)
            print("Mean reward achieved : {} in {} steps".format(np.mean(testing_rewards), np.mean(testing_steps)))
            print("-"*50)
            self.plot_rewards(save=True)
        self.plot_rewards(save=True)

    def plot_rewards(self, save=False):
        train_rewards = [self.train_rewards[i:i+self.test_freq] for i in range(0,len(self.train_rewards),self.test_freq)]
        test_rewards = [self.test_rewards[i:i+self.test_episodes] for i in range(0,len(self.test_rewards),self.test_episodes)]
        train_losses = [self.losses[i:i+self.test_freq] for i in range(0,len(self.losses),self.test_freq)]

        # rewards
        train_rewards_mean = [np.mean(i) for i in train_rewards]
        test_rewards_mean = [np.mean(i) for i in test_rewards]
        train_rewards_std = [np.std(i) for i in train_rewards]
        test_rewards_std = [np.std(i) for i in test_rewards]
        train_nepisodes = [self.test_freq * (i+1) for i in range(len(train_rewards_mean))]

        # steps
        train_steps = [self.train_steps[i:i+self.test_freq] for i in range(0,len(self.train_steps),self.test_freq)]
        test_steps = [self.test_steps[i:i+self.test_episodes] for i in range(0,len(self.test_steps),self.test_episodes)]
        train_steps_mean = [np.mean(i) for i in train_steps]
        test_steps_mean = [np.mean(i) for i in test_steps]
        train_steps_std = [np.mean(i) for i in train_steps]
        test_steps_std = [np.mean(i) for i in test_steps]

        # loss
        train_losses_mean = [np.mean(i) for i in train_losses]
        train_losses_std = [np.std(i) for i in train_losses]

        # training : reward over time
        plt.figure(1)
        plt.clf()
        plt.title("Training : Avg. Reward over {} episodes".format(self.test_episodes))
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg Reward")
        plt.errorbar(train_nepisodes, train_rewards_mean, yerr=train_rewards_std, color="indigo", uplims=True, lolims=True)
        if save :
            plt.savefig(self.expt_name + "train_rewards_{}.png".format(len(self.train_rewards)))
        else:
            plt.show()
            # pause so that the plots are updated
            plt.pause(0.001)

        # testing : reward over time
        plt.figure(2)
        plt.clf()
        plt.title("Testing : Avg. Reward over {} episodes".format(self.test_episodes))
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg Reward")
        try:
            plt.errorbar(train_nepisodes, test_rewards_mean, yerr=test_rewards_std, color="indigo", uplims=True, lolims=True)
        except:
            ipdb.set_trace()
        if save :
            plt.savefig(self.expt_name + "test_rewards_{}.png".format(len(self.test_rewards)))
        else:
            plt.show()

        # training : avg number of steps per episode
        plt.figure(3)
        plt.clf()
        plt.title("Training : Avg. number of steps taken per episode")
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg number of steps")
        plt.errorbar(train_nepisodes, train_steps_mean, yerr=train_steps_std, color="navy", uplims=True, lolims=True)
        if save :
            plt.savefig(self.expt_name + "train_steps_{}.png".format(len(self.train_steps)))
        else:
            plt.show()
            # pause so that the plots are updated
            plt.pause(0.001)

        # testing : avg number of steps per episode
        plt.figure(4)
        plt.clf()
        plt.title("Testing : Avg. number of steps taken per episode")
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg number of steps")
        plt.errorbar(train_nepisodes, test_steps_mean, yerr=test_steps_std, color="navy", uplims=True, lolims=True)
        if save :
            plt.savefig(self.expt_name + "test_steps_{}.png".format(len(self.test_steps)))
        else:
            plt.show()

        # training : avg loss over time
        plt.figure(5)
        plt.clf()
        plt.title("Avg. Training Loss over {} episodes".format(train_nepisodes[-1]))
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg. Loss")
        # plt.plot(train_losses_mean, color="crimson")
        plt.errorbar(train_nepisodes, train_losses_mean, yerr=train_losses_std, color="crimson", uplims=True, lolims=True)
        if save :
            plt.savefig(self.expt_name + "train_loss_{}.png".format(len(self.test_rewards)))
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Using REINFORCE for solving LunarLander")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor (default = 0.99)")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate (default = 1e-2)")
    parser.add_argument("--num_episodes", type=int, default=50000, help="number of episodes")
    parser.add_argument("--test_episodes", type=int, default=100, help="number of episodes to test on")
    parser.add_argument("--num_steps", type=int, default=50, help="number of steps to run per episode")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--save_freq", type=int, default=1e4, help="checkpoint frequency for saving models")
    parser.add_argument("--test_freq", type=int, default=500, help="frequency for testing policy")
    parser.add_argument("--save_path", type=str, default="models/reinforce2/", help="path for saving models")
    parser.add_argument("--expt_name", type=str, default="plots/reinforce2/", help="expt name for saving results")

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    # create the environment
    env = gym.make("LunarLander-v2")
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    # plt.ion()

    # REINFORCE agent
    agent = Reinforce(env, args)
    agent.train(args.num_episodes)
    # agent.test()

    env.close()



if __name__ == "__main__":
    main()
