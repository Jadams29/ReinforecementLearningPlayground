import os
import sys
import time
import gym
import slimevolleygym
import highway_env

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
"""


def create(environment_name):
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
    
    if environment_name not in all_environments:
        print("Incorrect Environment Specified")
        return None
    
    return
