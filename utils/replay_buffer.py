import os
import torch
import numpy as np
from joblib import Parallel, delayed

from utils.utils import to_numpy
from utils.sum_tree import SumTree, ParallelBatchSumTree

import pickle
import random
import sklearn.preprocessing


def take(array, start, end):
    """
    get array[start:end] in a circular fashion which turns out to be expensive...
    """
    # if start >= end:
    #     end += len(array)
    # idxes = np.arange(start, end) % len(array)
    # return array[idxes]
    return array[start:end]


def assign(array, start, end, value):
    if start >= end:
        end += len(array)
    idxes = np.arange(start, end) % len(array)
    array[idxes] = value


class ReplayBuffer:
    def __init__(self, params, file_path):
        # load in the data, split into train and test
        with open(file_path, "rb") as fp:  # Pickling
            data_list = pickle.load(fp)

        # random.shuffle(data_list, lambda: 0.5)

        if params.domain == "minigrid":
            scans, actions, rewards = data_list
            self.scans = np.array([a["image"] for a in scans])
            self.scans = np.transpose(self.scans, (0, 3, 1, 2))
            self.actions = np.array(actions)
            self.rewards = np.array(rewards)
            self.data_length = self.rewards.shape[0]
            if params.convert_data_onehot:
                #  a list of tensors, [(bs, num_pred_steps, feature_i_dim)] * feature_dim
                rewards_list = []
                for i in range(self.rewards.shape[1]):
                    label_binarizer = sklearn.preprocessing.LabelBinarizer()
                    unique_values = np.unique(self.rewards[:, i])
                    assert len(unique_values) <= params.reward_feature_inner_dim[i]
                    label_binarizer.fit(unique_values)
                    out = label_binarizer.transform(self.rewards[:, i])
                    if params.reward_feature_inner_dim[i] == 2:
                        out = np.hstack((out, 1 - out))
                    rewards_list.append(out)
                self.rewards = rewards_list

        if params.domain == "igibson":
            self.data_length = len(data_list)
            self.scans = np.array([a[0]["scan"] for a in data_list])
            self.scans = np.transpose(self.scans, (0, 2, 1))  # (total, 220, 1) - >  (total, 1, 220)
            print(f"scan shape: {self.scans.shape}")
            self.actions = np.array([a[1] for a in data_list])  # (total, action_dim)
            self.task_obs = np.array([a[0]["task_obs"] for a in data_list])

            if params.igibson_reward_type == "discrete":
                # shape: (total, 1)
                base_collision = np.array([a[2]["base_collision"] for a in data_list]).reshape(self.data_length, -1)
                arm_collision = np.array([a[2]["arm_collision"] for a in data_list]).reshape(self.data_length, -1)
                self_collision = np.array([a[2]["self_collision"] for a in data_list]).reshape(self.data_length, -1)

                self.rewards = np.concatenate([base_collision, arm_collision, self_collision], axis=1).astype(int)
                if params.convert_data_onehot:
                    #  a list of tensors, [(bs, num_pred_steps, feature_i_dim)] * feature_dim
                    rewards_list = []
                    for i in range(self.rewards.shape[1]):
                        reward_dim = self.rewards[:, i]
                        n_values = params.reward_feature_inner_dim[i]
                        rewards_list.append(np.eye(n_values)[reward_dim])
                    self.rewards = rewards_list
            elif params.igibson_reward_type == "continuous":
                self.rewards = np.array([np.append(a[3][:3],a[3][6:]) for a in data_list])  # (total, reward_dim)

        self.train_test_split = int(0.8 * self.data_length)
        self.device = params.device
        self.one_hot_reward = params.convert_data_onehot
        self.domain = params.domain


    def sample_inference(self, batch_size, use_part="all"):
        '''
        return: actions, obss, rewards
        - actions: (bs, 1, action_size)
        - obss: (bs, 1, 220, 1)
        - reward: (bs, 1, r_size)
        '''
        if use_part == "all":
            st_idx = 0
            ed_idx = self.data_length
        elif use_part == "train":
            st_idx = 0
            ed_idx = self.train_test_split
        elif use_part == "eval":
            st_idx = self.train_test_split
            ed_idx = self.data_length
        else:
            raise NotImplementedError

        idxs = np.random.choice(np.arange(st_idx, ed_idx), size=batch_size, replace=False).astype(int)
        actions, observations, rewards = self.sample_with_idx(idxs)
        return actions, observations, rewards, idxs

    def sample_with_idx(self, idxs):

        actions = self.actions[idxs]

        scans = self.scans[idxs]

        # Convert them to proper device & format
        # assumes: if self.continuous_action else torch.int64
        actions = torch.tensor(actions,
                               dtype=torch.float32 , device=self.device).unsqueeze(dim=1)
        scans = torch.tensor(scans,
                               dtype=torch.float32, device=self.device).unsqueeze(dim=1)
        if self.domain == "igibson":
            task_obs = self.task_obs[idxs]
            task_obs = torch.tensor(task_obs,
                               dtype=torch.float32, device=self.device).unsqueeze(dim=1)
            observations = [scans, task_obs]
        elif self.domain == "minigrid":
            observations = scans
        else:
            raise NotImplementedError

        if self.one_hot_reward:
            rewards = []
            for i in range(len(self.rewards)):
                reward_tensor = torch.tensor(self.rewards[i][idxs],
                                   dtype=torch.float32, device=self.device).unsqueeze(dim=1)
                rewards.append(reward_tensor)
        else:
            rewards = self.rewards[idxs]
            rewards = torch.tensor(rewards,
                                   dtype=torch.float32, device=self.device).unsqueeze(dim=1)

        return actions, observations, rewards


class ParallelPrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, params, file_path):
        self.capacity = capacity = params.training_params.replay_buffer_params.capacity
        self.reward_dim = params.reward_dim
        self.inference_batch_size = params.contrastive_params.batch_size
        self.inference_train_trees = ParallelBatchSumTree(self.reward_dim, capacity, self.inference_batch_size)

        self.alpha = params.training_params.replay_buffer_params.prioritized_alpha
        self.max_priority = 1
        self.num_observation_steps = params.training_params.num_observation_steps
        self.num_inference_pred_steps = params.contrastive_params.num_pred_steps

        super(ParallelPrioritizedReplayBuffer, self).__init__(params, file_path)

        assert (self.capacity >= self.data_length)

        # Only the train data has priority
        train_priorities = np.zeros(self.data_length)
        train_priorities[:self.train_test_split] = 1
        self.inference_train_trees.init_trees(train_priorities)

    def update_priorties(self, idxes, probs, type):
        if isinstance(probs, torch.Tensor):
            probs = to_numpy(probs)

        # probs = np.minimum(probs ** self.alpha, self.max_priority)
        probs = np.clip(probs ** self.alpha, 1e-4, self.max_priority)

        if type == "inference":
            trees = self.inference_train_trees
            # idxes, probs: (feature_dim, bs)
            trees.update(idxes, probs)
        else:
            raise NotImplementedError

    def sample_idxes_from_parallel_trees(self, batch_size, num_steps):
        # - self.max_priority * num_steps to avoid infinite loop of sampling the newly added sample
        trees = self.inference_train_trees
        segment = trees.total() / batch_size  # (feature_dim,)  # removed - self.max_priority * num_steps)

        s = np.random.uniform(size=(self.reward_dim, batch_size)) + np.arange(batch_size)
        s = s * segment[:, None]                            # (feature_dim, batch_size)

        # no need to validate idxes because we pre-set priorities of non-valid idxes to 0
        # tree_idxes, data_idxes: (feature_dim, batch_size)
        tree_idxes, data_idxes = trees.get(s)

        if np.any(data_idxes > 507999):
            tree_idxes, data_idxes = trees.get(s, monitor=True)
            exit()
            import sys
            sys.stdout = sys.__stdout__
            import ipdb
            ipdb.set_trace()

        data_idxes = np.array(data_idxes).flatten()         # (feature_dim * batch_size)
        return tree_idxes, data_idxes


    def sample_inference(self, batch_size, use_part="all"):
        num_steps = self.num_inference_pred_steps
        batch_size = self.inference_batch_size
        reward_dim = self.reward_dim

        if use_part == "train":
            # size: (reward_dim * batch_size)
            tree_idxes, data_idxes = self.sample_idxes_from_parallel_trees(batch_size, num_steps)
            actions, scans, rewards = self.sample_with_idx(data_idxes)
            actions_shape = actions.shape[2:]
            scans_shape = scans.shape[2:]
            rewards_shape = rewards.shape[2:]
            # TODO: reshape?   desired out: (reward_dim, bs, 1, n)
            actions = actions.reshape([reward_dim, batch_size, self.num_inference_pred_steps, *actions_shape])
            scans = scans.reshape([reward_dim, batch_size, self.num_inference_pred_steps, *scans_shape])
            rewards = rewards.reshape([reward_dim, batch_size, self.num_inference_pred_steps, *rewards_shape])

        else:
            actions, scans, rewards, tree_idxes = super(ParallelPrioritizedReplayBuffer, self).sample_inference(batch_size, use_part)

        return actions, scans, rewards, tree_idxes


