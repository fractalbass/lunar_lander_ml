import gym
import torch
from policy import Policy
from torch.autograd import Variable
from torch.distributions import Categorical
from model2_agent import Agent


class ModelAgent:

    # if gpu is to be used
    #use_cuda = torch.cuda.is_available()
    use_cuda = False
    gpu_id = 0
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
    Tensor = FloatTensor
    env = None
    agent = None

    def __init__(self):
        self.env = gym.make('LunarLander-v2')     
        self.agent = Agent(state_size=8, action_size=4, seed=3)
        self.policy.load_model("./models/network2.pth")
        print("Model has been loaded.")
        
    def land(self, filename=None):
        state = self.env.reset()
        steps = 0
        rewards = []
        states = []
        log_probs = []
        is_terminal = False
        
        while is_terminal==False:
            states.append(state)
            print(state)
            self.env.render()
            action, log_prob = self.select_action(state)
            print(action)
            state, reward, is_terminal, _ = self.env.step(action)
            log_probs.append(log_prob)
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
    for i in range(100):
        ma = ModelAgent()
        ma.land("mission_{}".format(i))

