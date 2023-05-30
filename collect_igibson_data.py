"""
Collect training data for causal inference
"""

import logging
import os

import igibson
from igibson.envs.igibson_env import iGibsonEnv
import yaml
from igibson.render.profiler import Profiler
import numpy as np
import pickle

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config_filename = os.path.join(igibson.configs_path, "fetch_reaching.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config_data["output"] = ['scan', 'task_obs']

    # Set task to factored version
    config_data["task"] = "factored_reaching_random"

    # Create a new environment for evaluation
    eval_env = iGibsonEnv(
        config_file=config_data,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
    )

    print(eval_env.action_space)
    max_iterations = 50000
    data_list = []
    for j in range(max_iterations):
        print("Resetting environment")
        prev_state = eval_env.reset()
        for i in range(100):
            with Profiler("Environment action step"):
                action = eval_env.action_space.sample()
                step_reward = None
                action_duration = 10
                collision_name_list = ["base_collision", "arm_collision", "self_collision", "collision_occur"]

                merged_info = {}
                for collision in collision_name_list:
                    merged_info[collision] = False

                for _ in range(action_duration):
                    state, reward, done, info = eval_env.step(action)
                    if step_reward is None:
                        step_reward = reward
                    else:
                        step_reward += reward

                    # Process info: has collision occur in the past n timesteps？
                    for collision_name in collision_name_list:
                        merged_info[collision_name] = merged_info[collision_name] or info[collision_name]

                    if done:
                        break

                # after action finished, store prev_state
                data = [prev_state, action, merged_info, step_reward]
                prev_state = state
                data_list.append(data)

                # Reset after collision
                if merged_info["collision_occur"]:
                    done = True

                if done:
                    print("Episode finished after {} timesteps".format(i + 1))
                    break

        if (j+1) % 500 == 0:
            print("\nsaving...\n")
            with open("causal_data", "wb") as fp:  # Pickling
                pickle.dump(data_list, fp)
    eval_env.close()



