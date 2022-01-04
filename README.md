# Time Series Classification

The goal of this project is to demonstrate how matrix profiles can be used to classify activity.  The context for this exploration will be the game lunar lander.  In this game, a player attempts to navigate and safely land a simulated lunar lander.  For the purposes of this experiment, the lunar lander will either be controlled by a person or by an autonomous agent.  Landings can further be classified as either successful or unsuccessful.  As a result, we will have a 2x2 classification problem.

Various models will be trained on the timeseries data of the lunar lander position with a resolution of 1/10 of a second.

This project will conclude with an evaluation of the accuracy of different modeling types for this classification problem.

Project Steps:

1.  Get the opengym lunar lander code to work 
2.  Get the keyboard_agent to work.
3.  Set up the recording of the (x,y) timestamp location on landing.
4.  Create a ML agent (reinforcement learning) to learn to train the lander.
5.  Compile a dataset that has both user and reinforcement learning examples of landing the lander
6.  Develop models that evaluate the landing.  (Matrix profile, xgboost, svm? other?)
7.  Write up results and create blog post.

Setup:

## Install swig:

sudo apt install swig

## Install pybox2d
pip install box2d-py

## Install openai gym:

pip install gym[Box2D]

# Run lunar langer with random agent.

<code>
import gym

env = gym.make('LunarLander-v2')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
</code>

To run with keyboard, use the keyboard_agent.py app in the examples/agents folder.  Then do:

<code>
python keyboard_agent.py
</code>

I followed these steps to install GPU nvidia cudnn/cuda stuff:

https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1

Train a reinforcement model with

<code>
python reinforce.py
</code>

