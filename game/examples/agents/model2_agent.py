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
        self.agent.network.load_state_dict(torch.load("./models/network2_500.pth"))

        self.env = gym.make('LunarLander-v2')

    def land(self, mission=None):
        state = self.env.reset()
        steps = 0
        rewards = []
        states = []
        log_probs = []
        is_terminal = False
        
        while is_terminal==False and steps < 800:
            print(state)
            self.env.render()
            action = self.agent.act(state)
            print(action)
            state, reward, is_terminal, _ = self.env.step(action)            
            states.append(state)
            rewards.append(reward)
            steps +=1   

        if mission!=None:
            filename = "./data/computer_missions_500/mission_{}_computer.csv".format(mission)
            with open(filename, "w") as fn:
                for state in states:
                    ln = ",".join([str(obs) for obs in state])
                    fn.writelines("{}\n".format(ln))  

        print("Landed!")
        self.env.close()

#  Models stuff.
    def select_action(self, state):
        state = Variable(self.Tensor(state))
        action_probs = self.policy(state)
        log_probs = action_probs.log()
        action = Categorical(action_probs).sample()
        return action.data.cpu().numpy(), log_probs[action]


if __name__ == "__main__":
    for i in range(1000):
        ma = Model2Agent()
        ma.land(i)