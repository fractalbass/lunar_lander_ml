# Time Series Classification using Matrix Profiles

The goal of this project is to demonstrate how matrix profiles can be used to classify activity.  The context for this exploration will be the game lunar lander.  In this game, a player attempts to navigate and safely land a simulated lunar lander.  For the purposes of this experiment, the lunar lander will either be controlled by a person or by an autonomous agent.  Landings can further be classified as either successful or unsuccessful.  As a result, we will have a 2x2 classification problem.

Various models will be trained on the timeseries data of the lunar lander position with a resolution of 1/10 of a second.

This project will conclude with an evaluation of the accuracy of different modeling types for this classification problem.

Project Steps:

1.  Get the pygame lunar lander code to work and record the (x,y) location on landing.
2.  Extend the pygame lunar lander code to also look at 1 second after impact?
3.  Get the pygame lunar lander to save the timeseries data for the lander code.
4.  Create a ML model (reinforcement learning) to learn to train the lander.
5.  Compile a dataset that has both user and reinforcement learning examples of landing the lander
6.  Develop models that evaluate the landing.  (Matrix profile, xgboost, svm? other?)
7.  Write up results and create blog post.

