import os
import sys
import time
import gym
import slimevolleygym
import highway_env

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 500
mpl.rcParams["agg.path.chunksize"] = 1000

plt.tight_layout()

"""
Environments:
    Atari: https://gym.openai.com/envs/#atari
        Breakout-ram-v0
        Breakout-v0
        Enduro-ram-v0
        Enduro-v0
        KungFuMaster-ram-v0
        KungFuMaster-v0
        MsPacman-ram-v0
        MsPacman-v0
        Pong-ram-v0
        Pong-v0
        Skiing-ram-v0
        Skiing-v0
        VideoPinball-ram-v0
        VideoPinball-v0
        
    Box2D: https://gym.openai.com/envs/#box2d
        BipedalWalker-v2
        BipedalWalkerHardcore-v2
        CarRacing-v0
        LunarLander-v2
        LunarLanderContinuous-v2
    
    Classic Control: https://gym.openai.com/envs/#classic_control
        Acrobot-v1
        CartPole-v1
        MountainCar-v0
        MountainCarContinuous-v0
        Pendulum-v0
    
    Robotics: https://gym.openai.com/envs/#roboticsac
        FetchPickAndPlace-v1
        FetchPush-v1
        FetchReach-v1
        FetchSlide-v1
        HandManipulateBlock-v0
        HandManipulateEgg-v0
        HandManipulatePen-v0ac
        HandReach-v0
    
    ToyText: https://gym.openai.com/envs/#toy_text
        FrozenLake-v0
        FrozenLake8x8-v0
        Taxi-v3
    
    3rd Party Environments: https://github.com/openai/gym/blob/master/docs/environments.md#third-party-environments
    
        Highway: import highway_env    https://github.com/eleurent/highway-env
            highway-v0
            merge-v0
            roundabout-v0
            parking-v0
            intersection-v0
            
        Slime Volleyball Gym (Multi-Agent): import slimevolleygym    https://github.com/hardmaru/slimevolleygym
            SlimeVolley-v0
            SlimeVolleyPixel-v0
            SlimeVolleyNoFrameskip-v0
        
        Goddard Rocket: https://github.com/osannolik/gym-goddard
            gym_goddard:Goddard-v0
    
s
"""

if __name__ == '__main__':
    # parser = argparse.ArgumentParser("Project 3 - Agent")
    # parser.add_argument("--algorithm", action='store', dest="algorithm", default="All")
    # parser.add_argument("--iterations", action='store', dest="numberOfIterations", default=10000)
    # parser.add_argument("--grid", action='store', dest="grid", default='Small')
    # args = parser.parse_args()
    # algs = {"QLearner", "FriendQ", "FoeQ", "CEQ", "All"}
    # if args.algorithm not in algs:
    #     print(f"Incorrect algorithm passed to script\n Valid options are {algs}")
    #     print("Exiting...")
    #     exit()
    # if args.grid == 'Small':
    #     myAgent = Agent(simulateActions=False, useSmallGrid=True, iterations=int(args.numberOfIterations),
    #                     algorithm=args.algorithm)
    #     myAgent.runTraining()
    # elif args.grid == 'Large':
    #     myAgent = Agent(simulateActions=False, useSmallGrid=False, iterations=int(args.numberOfIterations),
    #                     algorithm=args.algorithm)
    #     myAgent.runTraining()
    # elif args.grid == 'Both':
    #     myAgent = Agent(simulateActions=False, useSmallGrid=True,
    #                     iterations=int(args.numberOfIterations), algorithm=args.algorithm)
    #     myAgent.runTraining()
    #     myAgent = Agent(simulateActions=False, useSmallGrid=False,
    #                     iterations=int(args.numberOfIterations), algorithm=args.algorithm)
    #     myAgent.runTraining()

    all_environments = {'Breakout-ram-v0', 'Breakout-v0', 'Enduro-ram-v0', 'Enduro-v0', 'KungFuMaster-ram-v0',
                        'KungFuMaster-v0', 'MsPacman-ram-v0', 'MsPacman-v0', 'Pong-ram-v0', 'Pong-v0', 'Skiing-ram-v0',
                        'Skiing-v0', 'VideoPinball-ram-v0', 'VideoPinball-v0',

                        'BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'CarRacing-v0', 'LunarLander-v2',
                        'LunarLanderContinuous-v2',

                        'Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0',

                        'FrozenLake-v0', 'FrozenLake8x8-v0', 'Taxi-v3',

                        'highway-v0', 'merge-v0', 'roundabout-v0', 'parking-v0', 'intersection-v0',
                        'SlimeVolley-v0', 'SlimeVolleyPixel-v0', 'SlimeVolleyNoFrameskip-v0',
                        'gym_goddard:Goddard-v0'}
    
    for i in all_environments:
        try:
            env = gym.make(i)
        except Exception as _exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Exception in ''", _exception)
    print()
