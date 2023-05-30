import logging
import os
from typing import Callable

import igibson
from igibson.envs.igibson_env import iGibsonEnv

try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO, FPPO, A2C, SAC, TD3
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.save_util import load_from_zip_file
    from stable_baselines3.common.callbacks import CheckpointCallback


except ModuleNotFoundError:
    print("stable-baselines3 is not installed. ")
    exit(1)

import argparse
import uuid
import yaml
import numpy as np

"""
Main training code
"""


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            if key in ["proprioception", "task_obs"]:
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU())
            elif key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
                n_input_channels = subspace.shape[2]  # channel last
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key in ["scan"]:
                n_input_channels = subspace.shape[1]  # channel last
                cnn = nn.Sequential(
                    nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[1], subspace.shape[0]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            else:
                raise ValueError("Unknown observation key: %s" % key)
            total_concat_size += feature_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
                observations[key] = observations[key].permute((0, 3, 1, 2))
            elif key in ["scan"]:
                observations[key] = observations[key].permute((0, 2, 1))
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

def get_causal_matrix(reward_channels_dim, env, robot="fetch", fc_causal=False, sparse_causal=False):
    if fc_causal:
        causal_matrix = th.ones(reward_channels_dim, env.action_space.shape[0])
    elif sparse_causal:
        # Sparse causal is the causal matrix we derived from CMI
        if robot == "fetch":
            causal_matrix = th.tensor([
                [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # Reach
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # EE Local Orientation
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # EE Local Position
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Base Collision
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Arm Collision
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],  # Self Collision
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Head Attention
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Gripper Grasp
            ], dtype=th.float32)
        elif robot == "hsr":
            # base (3), head (2), arm (5), gripper (1)
            causal_matrix = th.tensor([
                #|omni  ,|head,|      arm    ,|gr
                [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # Reach
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # EE Local Orientation
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # EE Local Position
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Base Collision
                [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # Arm Collision
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],  # Self Collision
                [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # Head Attention
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Gripper Grasp
            ], dtype=th.float32)
    else:
        # We test assigning causality based on base / arm separation
        if robot == "fetch":
            # base (3), head (2), arm (5), gripper (1)
            causal_matrix = th.tensor([
                #|omni|head |      arm    ,   |gr
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Reach
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # EE Local Orientation
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # EE Local Position
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Base Collision
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Arm Collision
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # Self Collision
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Head Attention
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Gripper Grasp
            ], dtype=th.float32)
        elif robot == "hsr":
            # base (3), head (2), arm (5), gripper (1)
            causal_matrix = th.tensor([
                # |omni  ,|head,|      arm    ,|gr
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Reach
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # EE Local Orientation
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # EE Local Position
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Base Collision
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Arm Collision
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # Self Collision
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Head Attention
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Gripper Grasp
            ], dtype=th.float32)

    print(f"causal_matrix: {causal_matrix}")
    assert (causal_matrix.shape == (reward_channels_dim, env.action_space.shape[0]))
    return causal_matrix

def main(args, simple_test=False):
    """
    Train a RL agent using selected algorithm
    on a multi-step reaching task
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + '\n' + "*" * 80)

    # Check string argument
    assert args.robot in ["fetch", "hsr"]
    assert args.algo_name in ["ppo", "sac", "a2c", "td3", "fppo"]

    if args.robot == "fetch":
        config_name = "fetch_reaching.yaml"
    elif args.robot == "hsr":
        config_name = "hsr_reaching.yaml"

    config_filename = os.path.join(igibson.configs_path, config_name)
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    device = th.device("cuda:" + str(args.cuda_id) if th.cuda.is_available() else "cpu")

    # Parse the arguments
    short_exec = args.short
    factored = args.factored
    sep_value = args.seperate_v_net
    rd_target = args.random_target
    load_obstacles = args.load_obstacles

    if factored:
        args.algo_name = "fppo"
        if args.multi_step:
            config_data["task"] = "factored_multistep_reaching_random"
        else:
            config_data["task"] = "factored_reaching_random"
        print("using factored environment")
    else:
        if args.multi_step:
            config_data["task"] = "multistep_reaching_random"

    # We only log the important entrees, for simplicity
    log_names = {}
    log_parm_names = {
        "algoName": args.algo_name,
        "robotName": args.robot,
    }
    if "ppo" in args.algo_name:
        log_names = {
            "NormAD": args.normalize_advantage,
            "FcCausal": args.fc_causal,
            "SparseCausal": args.sparse_causal,
        }

    log_names["randomEnv"] = args.rand_env
    log_names["complexOri"] = not args.simple_orientation

    # Task specific arguments
    if not load_obstacles:
        config_data["load_room_types"] = "kitchen"
    config_data["rd_target"] = rd_target
    config_data["simple_orientation"] = args.simple_orientation
    config_data["enum_orientation"] = args.enum_orientation
    config_data["position_reward"] = not args.no_local_pos_reward
    config_data["proportional_local_reward"] = args.proportional_local_reward

    tensorboard_log_dir = "log_dir"
    log_base = uuid.uuid4().hex.upper()[:5]  # Each log has a unique identifier
    for k, v in log_parm_names.items():
        if v is not None:
            log_base = log_base + "_" + k + ":" + str(v)
    for k, v in log_names.items():
        if v:
            log_base = log_base + "_" + k
    weight_dir = os.path.join("weight_dir", log_base)
    num_environments = 8 if not short_exec else 2

    scene_id_list = ["Rs_int", "Beechwood_0_int", "Merom_0_int", "Wainscott_0_int",
                     "Ihlen_0_int", "Benevolence_1_int", "Pomaria_1_int", "Ihlen_1_int", ]

    # Function callback to create environments
    def make_env(rank: int, seed: int = 0, rand_env=False) -> Callable:
        if rand_env:
            cur_config = config_data.copy()
            cur_config["scene_id"] = scene_id_list[rank]
        else:
            cur_config = config_data

        def _init() -> iGibsonEnv:
            env = iGibsonEnv(
                config_file=cur_config,
                mode="headless",
                action_timestep=1 / 10.0,
                physics_timestep=1 / 120.0,
            )
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    # generate a random seed
    seed = np.random.randint(200000000)

    # Multiprocess
    env = SubprocVecEnv([make_env(i, seed, args.rand_env) for i in range(num_environments)])
    env = VecMonitor(env)

    # Create a new environment for evaluation
    eval_env = iGibsonEnv(
        config_file=config_data,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
    )

    # Obtain the arguments/parameters for the policy and create the PPO model
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    if simple_test:
        print("Perfoming simple testing")

        start_n_eval = 0
        final_n_eval = 1
        if "ppo" in args.algo_name:
            kwargs = {'n_steps': 16}
        elif args.algo_name == "a2c":
            kwargs = {'n_steps': 16}
        else:
            kwargs = {}
    else:
        kwargs = {}
        start_n_eval = 10
        final_n_eval = 20

    if "ppo" in args.algo_name:
        kwargs["clip_range"] = args.clip_range
        kwargs["target_kl"] = args.target_kl
        if not args.normalize_advantage:
            kwargs["normalize_advantage"] = False
        kwargs["gae_lambda"] = args.gae_lambda
    else:
        print("PPO parameters are ignore due to not applicable")

    # learning rate can either be a constant or a schedule
    if args.scheduled_lr:
        def linear_schedule(initial_value: float) -> Callable[[float], float]:
            """
            Linear learning rate schedule.

            :param initial_value: Initial learning rate.
            :return: schedule that computes
              current learning rate depending on remaining progress
            """

            def func(progress_remaining: float) -> float:
                """
                Progress will decrease from 1 (beginning) to 0.

                :param progress_remaining:
                :return: current learning rate
                """
                return progress_remaining * initial_value

            return func
        kwargs["learning_rate"] = linear_schedule(args.learning_rate)
    else:
        kwargs["learning_rate"] = args.learning_rate

    if factored:
        # Factored default to FPPO
        reward_channels_dim = 8
        causal_matrix = get_causal_matrix(reward_channels_dim, env, robot=args.robot,
                                          fc_causal=args.fc_causal,
                                          sparse_causal=args.sparse_causal)
        kwargs["sep_vnet"] = sep_value
        kwargs["value_loss_normalization"] = args.normalize_vnet_error
        kwargs["value_grad_rescale"] = args.rescale_vnet_grad
        kwargs["approx_var_gamma"] = args.normalize_vnet_error
        kwargs["episode_length"] = config_data["max_step"]
        model = FPPO("MultiInputPolicy", env, reward_channels_dim, causal_matrix, verbose=1,
                     tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs,
                     device=device, **kwargs)
    elif args.algo_name == "ppo":
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir,
                    device=device, policy_kwargs=policy_kwargs, **kwargs)
    elif args.algo_name == "sac":
        model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir,
                    device=device, policy_kwargs=policy_kwargs, **kwargs)
    elif args.algo_name == "a2c":
        model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir,
                    device=device, policy_kwargs=policy_kwargs, **kwargs)
    elif args.algo_name == "td3":
        model = TD3("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir,
                    device=device, policy_kwargs=policy_kwargs, **kwargs)

    print(model.policy)

    if start_n_eval > 0:
        # Random Agent, evaluation before training
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=start_n_eval)
        print(f"Before Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

    # Information related to storing weights
    # Save a checkpoint every 1000 steps
    model_path = os.path.join(weight_dir, "ckpt")
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=weight_dir,
        name_prefix="ckpt",
    )

    # Train the model for the given number of steps
    total_timesteps = 100 if short_exec else args.training_length
    model.learn(total_timesteps, tb_log_name=log_base, callback=checkpoint_callback)

    # Evaluate the policy after training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=final_n_eval)
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

    # Save the trained model and delete it
    model.save(model_path)
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL agent for iGibson.')

    parser.add_argument('--short', '-s', action='store_true',
                        help='whether to execute short')
    parser.add_argument('--seperate_v_net', '-sv', action='store_true',
                        help='whether to seperate the value network')
    parser.add_argument('--fc_causal', '-fc', action='store_true')
    parser.add_argument('--sparse_causal', '-sc', action='store_true')
    parser.add_argument('--cuda_id', '-cid', type=int, default=0)
    parser.add_argument('--clip_range', '-cr', type=float, default=0.2)
    parser.add_argument('--target_kl', '-tkl', type=float, default=0.15)
    parser.add_argument('--scheduled_lr', '-slr', action='store_true')
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5)
    parser.add_argument('--training_length', '-tr', type=int, default=5000000)
    parser.add_argument('--gae_lambda', '-gl', type=float, default=0.95)
    parser.add_argument('--robot', type=str)
    parser.add_argument('--normalize_vnet_error', '-nve', action='store_true')
    parser.add_argument('--rescale_vnet_grad', '-rvg', action='store_true')
    parser.add_argument('--no_local_pos_reward', '-npr', action='store_true',
                        help='whether to disable the local position reward')
    parser.add_argument('--algo_name', type=str, default="ppo",
                        help='which baseline algorithm to use, if we are not using factored')
    parser.add_argument('--rand_env', action='store_true')

    # For now, these arguments are always true
    parser.add_argument('--factored', '-f', action='store_true',
                        help='whether to factorize action space')
    parser.add_argument('--load_obstacles', '-obst', action='store_true')
    parser.add_argument('--random_target', '-rdt', action='store_true',
                        help='whether to have random target for the local eef')
    parser.add_argument('--normalize_advantage', '-nad', action='store_true')
    parser.add_argument('--simple_orientation', '-sor', action='store_true',
                        help='whether to simplify the orientation penalty')
    parser.add_argument('--multi_step', '-mts', action='store_true',
                        help='whether to use the multi-step environment')
    parser.add_argument('--enum_orientation', '-eor', action='store_true',
                        help='whether to limit orientation range')
    parser.add_argument('--proportional_local_reward', '-plr', action='store_true',)

    args = parser.parse_args()

    # We make these arguments always True
    args.load_obstacles = args.random_target = args.normalize_advantage = args.simple_orientation = args.multi_step = \
        args.enum_orientation = args.proportional_local_reward = args.factored = True

    logging.basicConfig(level=logging.INFO)

    simple_test = th.cuda.device_count() < 2
    main(args, simple_test)
