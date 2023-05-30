import logging
import os
from typing import Callable
import numpy as np

import igibson
from igibson.envs.igibson_env import iGibsonEnv
import yaml
from igibson.render.profiler import Profiler
try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3 import FPPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.save_util import load_from_zip_file

except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)

from train import CustomCombinedExtractor, get_causal_matrix

"""
This is to test (and visualize) the trained policy
"""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=2)
    robot = "hsr"  # "fetch" #
    if robot == "hsr":
        config_fn = "hsr_reaching.yaml"
        model_path = "weight_dir/ckpt_hsr.zip"
    elif robot == "fetch":
        config_fn = "fetch_reaching.yaml"
        model_path = "weight_dir/ckpt_fetch.zip"
    config_filename = os.path.join(igibson.configs_path, config_fn)
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Improving visuals in the example (optional)
    config_data["enable_shadow"] = True
    config_data["enable_pbr"] = True

    scnene_ids = ["Rs_int", "Beechwood_0_int", "Merom_0_int",
                  "Wainscott_0_int", "Ihlen_0_int", "Benevolence_1_int", "Pomaria_1_int", "Ihlen_1_int", ]

    # No living: Benevolence_0_int, Beechwood_1_int, Benevolence_2_int, Pomaria_0_int
    #  3: really hard & large env  4: uninteresting env, 5: hard env, 6: empty, 7: with sofa but easy
    config_data["scene_id"] = scnene_ids[1]

    factored = True
    obstacles = True
    multi_step = True

    if factored:
        if multi_step:
            config_data["task"] = "factored_multistep_reaching_random"
        else:
            config_data["task"] = "factored_reaching_random"
    else:
        if multi_step:
            config_data["task"] = "factored_multistep_reaching_random"

    config_data["rd_target"] = True
    config_data["vis_ee_target"] = False
    config_data["simple_orientation"] = True
    config_data["enum_orientation"] = True
    config_data["position_reward"] = True
    config_data["proportional_local_reward"] = True

    if not obstacles:
        config_data["load_room_types"] = "kitchen"

    # Create a new environment for evaluation
    eval_env = iGibsonEnv(
        config_file=config_data,
        mode="gui_interactive",  # "headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
        print_reward=True,
        # use_pb_gui=True,
    )

    # Alternatively we can make the causal argument optional and store them in data -- this is probably a better way
    device = th.device("cpu")


    data, params, pytorch_variables = load_from_zip_file(
        model_path,
        device=device,
    )

    reward_channels_dim = 8
    causal_matrix = get_causal_matrix(reward_channels_dim, eval_env, robot=robot, fc_causal=True)
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    if factored:
        model = FPPO("MultiInputPolicy", eval_env, reward_channels_dim, causal_matrix,
                     policy_kwargs=policy_kwargs)

    else:
        model = PPO("MultiInputPolicy", eval_env,
                    policy_kwargs=policy_kwargs)

    model.set_parameters(params, exact_match=True, device=device)

    from datetime import datetime
    set_random_seed(int(datetime.now().timestamp()))

    # Evaluate the trained model loaded from file
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=False,
                                              reward_channels_dim=reward_channels_dim, report_factored_reward=True)

    print(f"After Loading: Mean reward: {mean_reward} +/- {std_reward}")
    print(f"After Loading: Mean reward: {mean_reward.sum()}")

