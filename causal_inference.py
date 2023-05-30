"""
Supports minigrid and igibson inference
- sparse minigrid has sparse reward and a single reward term
- full minigrid has 5 reward terms
- Igibson-discrete has 3 reward terms and 11 action dimensions
- Igibson-continuous has 5 reward terms and 11 action dimensions
"""

import os
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

from model.inference_cmi import InferenceCMI
# from model.contrastive_cmi import ContrastiveCMI

from utils.utils import TrainingParams, Logger, set_seed_everywhere, get_start_step_from_model_loading
from utils.replay_buffer import ReplayBuffer, ParallelPrioritizedReplayBuffer
from utils.plot import plot_adjacency_intervention_mask

def train(params):
    device = torch.device("cuda:{}".format(params.cuda_id) if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() < 2:
        device = "cpu"
    if params.domain == "minigrid":
        if params.mini_env_name == "sparse":
            params.action_part_dim = 4
            params.reward_dim = 1
            params.action_feature_inner_dim = [3, 3, 3, 3]
            params.reward_feature_inner_dim = [2]
            params.continuous_action = True
            params.continuous_reward = False
            params.convert_data_onehot = True
            params.obs_dim = 1
        elif params.mini_env_name == "full":
            params.action_part_dim = 4
            params.reward_dim = 5
            params.action_feature_inner_dim = [3, 3, 3, 3]
            params.reward_feature_inner_dim = [3, 3, 2, 2, 2]
            params.continuous_action = True  # Manually specify, since not loading from env
            params.continuous_reward = False
            params.convert_data_onehot = True
            params.obs_dim = 1
        else:
            raise NotImplementedError
    elif params.domain == "igibson":
        if params.igibson_reward_type == "discrete":
            params.action_part_dim = 11
            params.reward_dim = 3  # three collisions
            params.reward_feature_inner_dim = [2, 2, 2]
            params.continuous_action = True
            params.continuous_reward = False
            params.convert_data_onehot = True
            params.obs_dim = 2  # scan & task_obs
            params.obs_ind_dim = 20
        elif params.igibson_reward_type == "continuous":
            params.action_part_dim = 11
            params.reward_dim = 5  # except for the three collisions
            params.continuous_action = True
            params.continuous_reward = True
            params.convert_data_onehot = False
            params.obs_dim = 2  # scan & task_obs
            params.obs_ind_dim = 20

    if not params.continuous_reward:
        assert(len(params.reward_feature_inner_dim) == params.reward_dim)

    params.ind_action_dim = 1      # Max number of actions we are grouping together as an individual action
    params.ind_reward_dim = 1
    rb_path = os.path.join("data", params.rb_path)

    set_seed_everywhere(params.seed)

    params.device = device
    training_params = params.training_params
    replay_buffer_params = training_params.replay_buffer_params
    inference_params = params.inference_params
    contrastive_params = params.contrastive_params

    # init replay buffer
    use_prioritized_buffer = replay_buffer_params.prioritized_buffer
    if use_prioritized_buffer:
        replay_buffer = ParallelPrioritizedReplayBuffer(params, rb_path)
    else:
        replay_buffer = ReplayBuffer(params, rb_path)

    # init model
    encoder = None
    decoder = None

    inference_algo = params.training_params.inference_algo
    # For now, contrastive_cmi
    use_contrastive = "contrastive" in inference_algo
    if inference_algo == "cmi":
        Inference = InferenceCMI
    elif inference_algo == "contrastive_cmi":
        # Inference = ContrastiveCMI
        raise NotImplementedError
    else:
        raise NotImplementedError
    inference = Inference(encoder, decoder, params)

    start_step = get_start_step_from_model_loading(params)
    total_step = training_params.total_step
    num_inference_opt_steps = training_params.num_inference_opt_steps

    # init saving
    writer = None
    if num_inference_opt_steps or num_inference_opt_steps:
        writer = SummaryWriter(os.path.join(params.rslts_dir, "tensorboard"))
    model_dir = os.path.join(params.rslts_dir, "trained_models")
    os.makedirs(model_dir, exist_ok=True)

    for step in range(start_step, total_step):
        is_init_stage = step < training_params.init_step

        loss_details = {"inference": [],
                        "inference_eval": [],
                        "policy": []}

        # training and logging
        if is_init_stage:
            continue

        if num_inference_opt_steps > 0:
            inference_batch_size = contrastive_params.batch_size if use_contrastive else inference_params.batch_size
            inference.train()
            inference.setup_annealing(step)
            for i_grad_step in range(num_inference_opt_steps):
                action_batch, obss_batch, rewards_batch, idxes_batch = \
                    replay_buffer.sample_inference(inference_batch_size, "train")
                loss_detail = inference.update(action_batch, obss_batch, rewards_batch)

                if use_prioritized_buffer:
                    replay_buffer.update_priorties(idxes_batch, loss_detail["priority"], "inference")
                    if params.domain == "minigrid" and params.mini_env_name == "full":
                        if (step+1) % 1000 == 0:
                            reward_dim_we_care = 3  # only support 3 or 4
                            extracted_num = 3
                            priority_dim = loss_detail["priority"][reward_dim_we_care].cpu()
                            values, top_idx = torch.topk(priority_dim, extracted_num)
                            print(f"top priority for dim {reward_dim_we_care}: {values}")
                            top_batch_idx = idxes_batch[reward_dim_we_care][top_idx] - \
                                            params.training_params.replay_buffer_params.capacity + 1
                            print(f"center grid")
                            print(replay_buffer.scans[top_batch_idx][..., 0, 3, 3])  # This is the central grid
                            non_empty_neighbor = (replay_buffer.scans[top_batch_idx][..., 0, 2:5, 2:5] != 1
                                                  ).reshape(3, -1).sum(axis=1) % 3
                            print(f"ideal action: \n{non_empty_neighbor}")
                            print("actions:")
                            print(replay_buffer.actions[top_batch_idx][:, reward_dim_we_care-1])
                            print("rewards:")
                            print(replay_buffer.rewards[top_batch_idx][:, reward_dim_we_care])

                        if (step + 1) % 100 == 0:
                            reward_dim_we_care = 3  # only support 3 or 4
                            data_idxes = idxes_batch[reward_dim_we_care]  - params.training_params.replay_buffer_params.capacity + 1
                            special_r_list = replay_buffer.rewards[data_idxes][:, reward_dim_we_care] == -5
                            print(f"\n Number of special reward datapoint is {np.sum(special_r_list)} \n")

                loss_details["inference"].append(loss_detail)

            inference.eval()
            if (step + 1) % training_params.eval_freq == 0:
                if params.train_mask:
                    eval_data_part = "train"
                else:
                    eval_data_part = "eval"
                action_batch, obss_batch, rewards_batch, _ = \
                    replay_buffer.sample_inference(inference_batch_size, use_part=eval_data_part)
                loss_detail = inference.update(action_batch, obss_batch, rewards_batch, eval=True)
                loss_details["inference_eval"].append(loss_detail)
                print("{}/{}, init_stage: {}".format(step + 1, total_step, is_init_stage))
                cur_adj = inference.get_adjacency()[:, :-params.obs_dim]
                max_act = torch.max(cur_adj, dim=1)
                print(f"current adjacency: \n{cur_adj}")
                print(f"max action: {max_act}")
                print(f"current mask: \n{inference.get_mask()}")

        if writer is not None:
            for module_name, module_loss_detail in loss_details.items():
                if not module_loss_detail:
                    continue
                # list of dict to dict of list
                if isinstance(module_loss_detail, list):
                    keys = set().union(*[dic.keys() for dic in module_loss_detail])
                    module_loss_detail = {k: [dic[k].item() for dic in module_loss_detail if k in dic]
                                          for k in keys if k not in ["priority"]}
                for loss_name, loss_values in module_loss_detail.items():
                    writer.add_scalar("{}/{}".format(module_name, loss_name), np.mean(loss_values), step)

            if (step + 1) % training_params.plot_freq == 0 and num_inference_opt_steps > 0:
                plot_adjacency_intervention_mask(inference, writer, step)

        if (step + 1) % training_params.saving_freq == 0:
            if num_inference_opt_steps > 0:
                inference.save(os.path.join(model_dir, "inference_{}".format(step + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line arguments.')
    parser.add_argument('--config', type=str,
                        help='params to load from')
    args = parser.parse_args()
    params = TrainingParams(training_params_fname=args.config, train=True)
    train(params)
