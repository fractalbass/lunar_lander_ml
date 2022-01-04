import gym
import torch
from dqn_agent import Agent
from torch.autograd import Variable
from torch.distributions import Categorical

class Model2Agent:
    agent = None
    env = None

    def __init__(self):
        self.agent = Agent(state_size=8, action_size=4, seed=3)
        self.agent.network.load_state_dict(torch.load("./models/network2.pth"))

        self.env = gym.make('LunarLander-v2')

    def land(self):
        state = self.env.reset()
        steps = 0
        rewards = []
        log_probs = []
        is_terminal = False
        
        while is_terminal==False:
            print(state)
            self.env.render()
            action = self.agent.act(state)
            print(action)
            state, reward, is_terminal, _ = self.env.step(action)            
            rewards.append(reward)
            steps +=1            
        print("Landed!")

#  Models stuff.
    def select_action(self, state):
        state = Variable(self.Tensor(state))
        action_probs = self.policy(state)
        log_probs = action_probs.log()
        action = Categorical(action_probs).sample()
        return action.data.cpu().numpy(), log_probs[action]




if __name__ == "__main__":
    ma = Model2Agent()
    ma.land()