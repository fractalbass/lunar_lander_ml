import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, state_size, num_actions):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, num_actions)

    def load_model(self, model_name):
        torch.load(model_name, map_location=torch.device('cpu'))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x),dim=0)
        return x