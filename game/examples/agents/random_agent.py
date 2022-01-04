import gym

class RandomAgent:

    env = None

    def __init__(self):
        self.env = gym.make('LunarLander-v2')        
        
    def land(self):
        self.env.reset()
        landed = False
        while landed==False:
            self.env.render()
            state = self.env.step(self.env.action_space.sample())
            print("{}, {}".format(state[0][0], state[0][1]))
            if state[0][6]>0 or state[0][7]>0:
                landed=True
        print("Landed!")


if __name__ == "__main__":
    ra = RandomAgent()
    ra.land()
