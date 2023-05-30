import argparse
import numpy

from utils.utils import set_seed_everywhere
from utils.penv import ParallelEnv
import pickle
import numpy as np
import torch
import gym
import gym_minigrid

def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.reset(seed=seed)
    return env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--env", type=str, default="MiniGrid-SwampEnv-8x8-N3-v0",
                    help="Envrionment name")
parser.add_argument("--num", type=int, default=100000,
                    help="Number of datapoint to collect")
args = parser.parse_args()

# Set seed for all randomness sources
set_seed_everywhere(args.seed)

# Set device
print(f"Device: {device}\n")
procs = 8

# Load environment
envs = []
for i in range(procs):
    envs.append(make_env(args.env, args.seed + 10000 * i))
env = ParallelEnv(envs)
print("Environment loaded\n")

# Load agent
obs_list = []
rewards_list = []
actions_list = []

obs = env.reset()

save_fn = "minigrid_causal_data_" + args.env
num_of_data = args.num
for i in range(num_of_data):
    # Do one agent-environment interaction
    action = [env.action_space.sample() for _ in range(procs)]

    # For sparse reward scenario
    if "Sparse1d" in args.env:
        action_threshold = 0.7
        probs = np.random.uniform(size=procs)
        for j in range(procs):
            if probs[j] < action_threshold:
                action[j][0] = 2  # manually move towards target

    nxt_obs, reward, done, info = env.step(action)
    rewards_list += info  # info contains the decomposed reward
    obs_list += obs
    actions_list += action
    obs = nxt_obs

    if (i+1) % 500 == 0:
        print(f"saving iteration {i+1}...")
        with open(save_fn, "wb") as fp:  # Pickling
            pickle.dump([obs_list, actions_list, rewards_list], fp)





