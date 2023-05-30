raise NotImplementedError
# need to fix the obs dimension error
# grep -nrw "action_part_dim + 1"

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.contrastive import Contrastive
from model.inference_utils import reset_layer, forward_network, forward_network_batch

import ipdb


class ContrastiveCMI(Contrastive):
    def __init__(self, encoder, decoder, params):

        #initialize hard-coded variable
        self.action_part_dim = params.action_part_dim
        self.reward_dim = params.reward_dim

        self.cmi_params = params.contrastive_params.cmi_params
        self.init_graph(params, encoder)
        # TODO: change the init_graph
        super(ContrastiveCMI, self).__init__(encoder, decoder, params)
        self.aggregation = self.cmi_params.aggregation
        self.train_all_masks = self.cmi_params.train_all_masks
        self.mask_opt_freq = self.cmi_params.mask_opt_freq
        self.full_opt_freq = self.cmi_params.full_opt_freq
        self.causal_opt_freq = self.cmi_params.causal_opt_freq

        replay_buffer_params = params.training_params.replay_buffer_params
        self.parallel_sample = replay_buffer_params.prioritized_buffer and replay_buffer_params.parallel_sample

        self.update_num = 0



    def init_model(self):
        params = self.params
        cmi_params = self.cmi_params
        self.learn_bo = learn_bo = cmi_params.learn_bo
        self.dot_product_energy = dot_product_energy = cmi_params.dot_product_energy

        # model params
        continuous_state = self.continuous_state
        if not continuous_state:
            raise NotImplementedError

        # TODO
        #  This is a bit hacky: include a network.
        #  Another thing is that there should actually be n copies of these
        if params.domain == "igibson":
            lidar_shape = [220, 1]
            lidar_out = 128
            n_input_channels = lidar_shape[1]  # channel last
            self.obs_extractor = []
            for _ in range(params.reward_dim):
                cnn = nn.Sequential(
                    nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = torch.zeros([lidar_shape[1], lidar_shape[0]])
                with torch.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, lidar_out), nn.ReLU())
                self.obs_extractor.append(nn.Sequential(cnn, fc).to(params.device))
        elif params.domain == "minigrid":
            # Define image embedding
            img_shape = [3, 7, 7]
            img_out = 128
            self.obs_extractor = []
            for _ in range(params.reward_dim):
                cnn = nn.Sequential(
                    nn.Conv2d(3, 16, (2, 2)),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                    nn.Conv2d(16, 32, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, (2, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = torch.zeros(img_shape)
                with torch.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, img_out), nn.ReLU())
                self.obs_extractor.append(nn.Sequential(cnn, fc).to(params.device))
        else:
            raise NotImplementedError


        action_part_dim = self.action_part_dim
        reward_dim = self.reward_dim
        ar_dim = action_part_dim * reward_dim

        ### from here

        self.action_part_feature_weights = nn.ParameterList()
        self.action_part_feature_biases = nn.ParameterList()
        self.reward_feature_feature_weights = nn.ParameterList()
        self.reward_feature_feature_biases = nn.ParameterList()
        # self.delta_state_feature_weights = nn.ParameterList()
        # self.delta_state_feature_biases = nn.ParameterList()

        self.energy_weights = nn.ParameterList()
        self.energy_biases = nn.ParameterList()
        self.cond_energy_weights = nn.ParameterList()
        self.cond_energy_biases = nn.ParameterList()

        self.sa_encoder_weights = nn.ParameterList()
        self.sa_encoder_biases = nn.ParameterList()
        self.d_encoder_weights = nn.ParameterList()
        self.d_encoder_biases = nn.ParameterList()
        self.cond_sa_encoder_weights = nn.ParameterList()
        self.cond_sa_encoder_biases = nn.ParameterList()

        # state feature extractor
        in_dim = params.ind_action_dim * self.num_observation_steps
        for out_dim in cmi_params.feature_fc_dims:
            self.action_part_feature_weights.append(nn.Parameter(torch.zeros(ar_dim, in_dim, out_dim)))
            self.action_part_feature_biases.append(nn.Parameter(torch.zeros(ar_dim, 1, out_dim)))
            in_dim = out_dim

        # delta state feature extractor
        in_dim = params.ind_reward_dim
        for out_dim in cmi_params.feature_fc_dims:
            self.reward_feature_feature_weights.append(nn.Parameter(torch.zeros(reward_dim, in_dim, out_dim)))
            self.reward_feature_feature_biases.append(nn.Parameter(torch.zeros(reward_dim, 1, out_dim)))
            in_dim = out_dim

        if dot_product_energy:
            # sa_feature encoder
            in_dim = cmi_params.feature_fc_dims[-1]
            for out_dim in cmi_params.enery_fc_dims:
                self.sa_encoder_weights.append(nn.Parameter(torch.zeros(reward_dim, in_dim, out_dim)))
                self.sa_encoder_biases.append(nn.Parameter(torch.zeros(reward_dim, 1, out_dim)))
                in_dim = out_dim

            # delta feature encoder
            in_dim = cmi_params.feature_fc_dims[-1]
            for out_dim in cmi_params.enery_fc_dims:
                self.d_encoder_weights.append(nn.Parameter(torch.zeros(reward_dim, in_dim, out_dim)))
                self.d_encoder_biases.append(nn.Parameter(torch.zeros(reward_dim, 1, out_dim)))
                in_dim = out_dim

            # conditional sa_feature encoder
            if learn_bo:
                in_dim = 2 * cmi_params.feature_fc_dims[-1]
                for out_dim in cmi_params.enery_fc_dims:
                    self.cond_sa_encoder_weights.append(nn.Parameter(torch.zeros(reward_dim, in_dim, out_dim)))
                    self.cond_sa_encoder_biases.append(nn.Parameter(torch.zeros(reward_dim, 1, out_dim)))
                    in_dim = out_dim
        else:
            # energy
            in_dim = 2 * cmi_params.feature_fc_dims[-1]
            for out_dim in cmi_params.enery_fc_dims:
                self.energy_weights.append(nn.Parameter(torch.zeros(reward_dim, in_dim, out_dim)))
                self.energy_biases.append(nn.Parameter(torch.zeros(reward_dim, 1, out_dim)))
                in_dim = out_dim
            self.energy_weights.append(nn.Parameter(torch.zeros(reward_dim, in_dim, 1)))
            self.energy_biases.append(nn.Parameter(torch.zeros(reward_dim, 1, 1)))

            if learn_bo:
                # conditional energy
                in_dim = 3 * cmi_params.feature_fc_dims[-1]
                for out_dim in cmi_params.enery_fc_dims:
                    self.cond_energy_weights.append(nn.Parameter(torch.zeros(reward_dim, in_dim, out_dim)))
                    self.cond_energy_biases.append(nn.Parameter(torch.zeros(reward_dim, 1, out_dim)))
                    in_dim = out_dim
                self.cond_energy_weights.append(nn.Parameter(torch.zeros(reward_dim, in_dim, 1)))
                self.cond_energy_biases.append(nn.Parameter(torch.zeros(reward_dim, 1, 1)))

        # TODO
        training_masks = []
        for i in range(action_part_dim):
            training_masks.append(self.get_eval_mask((1,), i))

        # 1st feature_dim: variable to predict, 2nd feature_dim: input variable to ignore
        training_masks = torch.stack(training_masks, dim=2)  # (1, feature_dim, feature_dim, feature_dim + 1)
        self.training_masks = training_masks.view(reward_dim, action_part_dim, action_part_dim + 1, 1, 1)

    def reset_params(self):
        # feature_dim = self.feature_dim
        module_weights = [self.action_part_feature_weights,
                          self.reward_feature_feature_weights,
                          # self.delta_state_feature_weights,
                          self.energy_weights,
                          self.cond_energy_weights,
                          self.sa_encoder_weights,
                          self.d_encoder_weights,
                          self.cond_sa_encoder_weights]
        module_biases = [self.action_part_feature_biases,
                         self.reward_feature_feature_biases,
                         # self.delta_state_feature_biases,
                         self.energy_biases,
                         self.cond_energy_biases,
                         self.sa_encoder_biases,
                         self.d_encoder_biases,
                         self.cond_sa_encoder_biases]
        for weights, biases in zip(module_weights, module_biases):
            for w, b in zip(weights, biases):
                assert w.ndim == b.ndim == 3
                for i in range(w.shape[0]):
                    reset_layer(w[i], b[i])

    def init_graph(self, params, encoder):

        device = params.device
        self.CMI_threshold = self.cmi_params.CMI_threshold

        # feature_dim = encoder.feature_dim
        action_part_dim = self.action_part_dim
        reward_dim = self.reward_dim

        # TODO: Figure out diag_mask. and what happened in our experiment
        # # used for masking diagonal elements
        # self.diag_mask = torch.eye(reward_dim, action_part_dim + 1, dtype=torch.bool, device=device)
        self.diag_mask = torch.zeros(reward_dim, action_part_dim + 1, dtype=torch.bool, device=device)
        self.diag_mask[:, action_part_dim:] = True
        self.mask_CMI = torch.ones(reward_dim, action_part_dim + 1, device=device) * self.CMI_threshold
        self.mask = torch.ones(reward_dim, action_part_dim + 1, dtype=torch.bool, device=device)

    # modified from extract_action_feature
    def extract_observation_feature(self, obs):
        """
        Modified: takes in a ladar scan and output an obs feature
        (feature_dim) * bs * obs_dim
        """

        multi_bs = self.parallel_sample and self.training
        if not multi_bs:
            obs = obs.unsqueeze(dim=0)
            obs_shape = obs.shape[2:]
            obs_dim = len(obs_shape)
            dummy_axis = obs_dim * [-1]
            obs = obs.expand(self.reward_dim, -1, *dummy_axis)
        obs_features = []
        for i in range(self.reward_dim):
            obs_features.append(self.obs_extractor[i](obs[i]))

        obs_features = torch.stack(obs_features)

        dim_out = obs_features.shape[-1]
        obs_features = obs_features.reshape([self.reward_dim, 1, -1, dim_out])
        return obs_features

    # modified from extract_delta_state_feature
    def extract_reward_feature(self, rewards):
        """
        :param rewards:
            if state space is continuous: (bs, num_samples, reward_dim).
            else: [(bs, num_samples, feature_i_dim)] * feature_dim
            notice that bs must be 1D
        :return: (reward_dim, bs, num_samples, out_dim)
        """

        reward_dim = self.reward_dim
        if self.continuous_state:
            bs, num_samples, _ = rewards.shape
            x = rewards.view(-1, reward_dim).T               # (feature_dim, bs * num_samples)
            x = x.unsqueeze(dim=-1)                                 # (feature_dim, bs * num_samples, 1)
        else:
            raise NotImplementedError

        reward_feature = forward_network(x, self.reward_feature_feature_weights, self.reward_feature_feature_biases)
        reward_feature = reward_feature.view(reward_dim, bs, num_samples, -1)
        return reward_feature

    # modified from extract_state_feature
    def extract_action_part_feature(self, actions):
        """
        :param actions:
            if state space is continuous: (bs, num_observation_steps, feature_dim).
            else: [(bs, num_observation_steps, feature_i_dim)] * feature_dim
            notice that bs must be 1D
        :return: (feature_dim, feature_dim, bs, out_dim),
            the first feature_dim is each state variable at next time step to predict, the second feature_dim are
            inputs (all current state variables) for the prediction
        """
        action_part_dim = self.action_part_dim
        reward_dim = self.reward_dim

        if self.continuous_state:
            if self.parallel_sample and self.training:
                bs = actions.shape[1]
                x = actions.permute(0, 3, 1, 2)                    # (feature_dim, feature_dim, bs, num_observation_steps)
            else:
                bs = actions.shape[0]
                x = actions.permute(2, 0, 1).unsqueeze(dim=0)      # (1, action_part_dim, bs, num_observation_steps)
                x = x.repeat(reward_dim, 1, 1, 1)                  # (reward_dim, action_part_dim, bs, num_observation_steps)
            x = x.reshape(action_part_dim * reward_dim, bs, -1)        # (feature_dim * feature_dim, bs, 1)
        else:
            raise NotImplementedError

        actions_feature = forward_network(x, self.action_part_feature_weights, self.action_part_feature_biases)
        actions_feature = actions_feature.view(reward_dim, action_part_dim, bs, -1) # TODO
        return actions_feature

    @staticmethod
    def dot_product(sa_encoding, delta_encoding):
        """
        compute the dot product between sa_encoding and delta_encoding
        :param sa_encoding: (feature_dim, bs, encoding_dim) or (feature_dim, feature_dim, bs, encoding_dim),
            notice that bs must be 1D
        :param delta_encoding: (feature_dim, bs, num_samples, encoding_dim), global feature used for prediction,
            notice that bs must be 1D
        :return: energy: (bs, num_samples, feature_dim)
        """
        # (feature_dim, bs, 1, out_dim) or (feature_dim, feature_dim, bs, 1, out_dim)
        sa_encoding = sa_encoding.unsqueeze(dim=-2)

        if sa_encoding.ndim == 5:
            num_samples = delta_encoding.shape[-2]
            if num_samples < 5000:
                delta_encoding = delta_encoding.unsqueeze(dim=1)        # (feature_dim, 1, bs, num_samples, out_dim)
                energy = (sa_encoding * delta_encoding).sum(dim=-1)     # (feature_dim, feature_dim, bs, num_samples)
            else:
                # likely to have out of memory issue, so need to compute energy in batch
                energy = []
                for sa_encoding in torch.unbind(sa_encoding, dim=1):
                    energy.append((sa_encoding * delta_encoding).sum(dim=-1))
                energy = torch.stack(energy, dim=1)
            energy = energy.permute(2, 3, 0, 1)                         # (bs, num_samples, feature_dim, feature_dim)
        else:
            energy = (sa_encoding * delta_encoding).sum(dim=-1)         # (feature_dim, bs, num_samples)
            energy = energy.permute(1, 2, 0)                            # (bs, num_samples, feature_dim)

        return energy

    def compute_energy_dot(self, sa_feature, delta_feature, full_sa_feature=None):
        """
        compute the conditional energy from the conditional and total state-action-delta features
        :param sa_feature: (feature_dim, bs, sa_feature_dim) or (feature_dim, feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :param delta_feature: (feature_dim, bs, num_samples, delta_feature_dim), global feature used for prediction,
            notice that bs must be 1D
        :param full_sa_feature: (feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :return: energy: (bs, num_samples, feature_dim)
        """
        reward_dim, bs, num_samples, delta_feature_dim = delta_feature.shape
        action_part_dim = self.action_part_dim
        # (feature_dim, bs * num_samples, delta_feature_dim)
        delta_feature = delta_feature.view(reward_dim, bs * num_samples, -1)
        assert sa_feature.ndim in [3, 4]
        is_mask_feature = sa_feature.ndim == 4

        # (feature_dim, bs * num_samples, out_dim)
        delta_encoding = forward_network(delta_feature, self.d_encoder_weights, self.d_encoder_biases)
        # (feature_dim, bs, num_samples, out_dim)
        delta_encoding = delta_encoding.view(reward_dim, bs, num_samples, -1)

        if is_mask_feature:

            # (feature_dim, feature_dim * bs, sa_feature_dim)
            sa_feature = sa_feature.view(reward_dim, action_part_dim * bs, -1)

        # (feature_dim, bs, out_dim) or (feature_dim, feature_dim * bs, out_dim)
        sa_encoding = forward_network(sa_feature, self.sa_encoder_weights, self.sa_encoder_biases)

        if is_mask_feature:
            # (feature_dim, feature_dim, bs, out_dim)
            sa_encoding = sa_encoding.view(reward_dim, action_part_dim, bs, -1)

        # (bs, num_samples, feature_dim) or (bs, num_samples, feature_dim, feature_dim)
        energy = self.dot_product(sa_encoding, delta_encoding)

        if full_sa_feature is None:
            return energy

        if not self.learn_bo:
            return energy, torch.zeros_like(energy)

        if is_mask_feature:
            # (feature_dim, feature_dim, bs, out_dim)
            ipdb.set_trace()
            # TODO: this part needs to be double checked
            raise NotImplementedError
            full_sa_feature = full_sa_feature.unsqueeze(dim=1).expand(-1, action_part_dim, -1, -1)
            # (feature_dim, feature_dim * bs, out_dim)
            full_sa_feature = full_sa_feature.reshape(reward_dim, action_part_dim * bs, -1)
            # (feature_dim, feature_dim * bs, 2 * out_dim)
            cond_sa_feature = torch.cat([sa_feature, full_sa_feature], dim=-1)
        else:
            # (feature_dim, bs, 2 * out_dim)
            cond_sa_feature = torch.cat([sa_feature, full_sa_feature], dim=-1)

        # (feature_dim, bs, out_dim) or (feature_dim, feature_dim * bs, out_dim)
        cond_sa_encoding = forward_network(cond_sa_feature, self.cond_sa_encoder_weights, self.sa_encoder_biases)

        if is_mask_feature:
            # (feature_dim, feature_dim, bs, out_dim)
            cond_sa_encoding = cond_sa_encoding.view(reward_dim, action_part_dim, bs, -1)

        # (bs, num_samples, feature_dim) or (bs, num_samples, feature_dim, feature_dim)
        cond_energy = self.dot_product(cond_sa_encoding, delta_encoding)

        return energy, cond_energy

    @staticmethod
    def unsqueeze_expand_tensor(tensor, dim, expand_size):
        tensor = tensor.unsqueeze(dim=dim)
        expand_sizes = [-1] * tensor.ndim
        expand_sizes[dim] = expand_size
        tensor = tensor.expand(*expand_sizes)
        return tensor

    def net(self, sa_feature, delta_feature, weights, biases, full_sa_feature=None):
        is_mask_feature = sa_feature.ndim == 5
        sa_feature_dim = sa_feature.shape[-1]
        feature_dim, bs, num_samples, delta_feature_dim = delta_feature.shape

        if is_mask_feature and num_samples >= 1024:
            energy = []
            for sa_feature_i in torch.unbind(sa_feature, dim=1):
                if full_sa_feature is None:
                    sad_feature = torch.cat([sa_feature_i, delta_feature], dim=-1)
                else:
                    sad_feature = torch.cat([sa_feature_i, full_sa_feature, delta_feature], dim=-1)
                sad_feature_dim = sad_feature.shape[-1]

                # (feature_dim, bs * num_samples, sad_feature_dim)
                sad_feature = sad_feature.view(feature_dim, -1, sad_feature_dim)

                # (feature_dim, bs * num_samples, 1)
                energy_i = forward_network(sad_feature, weights, biases)
                energy.append(energy_i)
            energy = torch.stack(energy, dim=1)             # (feature_dim, feature_dim, bs * num_samples, 1)
            energy = energy.view(feature_dim, -1, 1)
        else:
            if is_mask_feature:
                # (feature_dim, feature_dim, bs, num_samples, delta_feature_dim)
                delta_feature = self.unsqueeze_expand_tensor(delta_feature, 1, feature_dim)

            if full_sa_feature is None:
                sad_feature = torch.cat([sa_feature, delta_feature], dim=-1)
            else:
                if is_mask_feature:
                    full_sa_feature = self.unsqueeze_expand_tensor(full_sa_feature, 1, feature_dim)
                sad_feature = torch.cat([sa_feature, full_sa_feature, delta_feature], dim=-1)

            sad_feature_dim = sad_feature.shape[-1]

            # (feature_dim, bs * num_samples, sad_feature_dim) or 
            # (feature_dim, feature_dim * bs * num_samples, sad_feature_dim)
            sad_feature = sad_feature.view(feature_dim, -1, sad_feature_dim)

            # (feature_dim, bs * num_samples, 1) or (feature_dim, feature_dim * bs * num_samples, 1)
            energy = forward_network(sad_feature, weights, biases)

        if is_mask_feature:
            energy = energy.view(feature_dim, feature_dim, bs, num_samples)
            energy = energy.permute(2, 3, 0, 1)                         # (bs, num_samples, feature_dim, feature_dim)
        else:
            energy = energy.view(feature_dim, bs, num_samples)          # (feature_dim, bs, num_samples)
            energy = energy.permute(1, 2, 0)                            # (bs, num_samples, feature_dim)

        return energy

    def compute_energy_net(self, sa_feature, delta_feature, full_sa_feature=None):
        """
        compute the conditional energy from the conditional and total state-action-delta features
        :param sa_feature: (feature_dim, bs, sa_feature_dim) or (feature_dim, feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :param delta_feature: (feature_dim, bs, num_samples, delta_feature_dim), global feature used for prediction,
            notice that bs must be 1D
        :param full_sa_feature: (feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :return: energy: (bs, num_samples, feature_dim) or (bs, num_samples, feature_dim, feature_dim)
        """

        assert sa_feature.ndim in [3, 4]
        is_mask_feature = sa_feature.ndim == 4

        sa_feature_dim = sa_feature.shape[-1]
        feature_dim, bs, num_samples, delta_feature_dim = delta_feature.shape
        
        # (feature_dim, bs, num_samples, sa_feature_dim) or (feature_dim, feature_dim, bs, num_samples, sa_feature_dim)
        sa_feature = self.unsqueeze_expand_tensor(sa_feature, -2, num_samples)

        # (bs, num_samples, feature_dim) or (bs, num_samples, feature_dim, feature_dim)
        energy = self.net(sa_feature, delta_feature, self.energy_weights, self.energy_biases)

        if full_sa_feature is None:
            return energy

        if not self.learn_bo:
            return energy, torch.zeros_like(energy)

        # (feature_dim, bs, num_samples, sa_feature_dim) or (feature_dim, feature_dim, bs, num_samples, sa_feature_dim)
        full_sa_feature = self.unsqueeze_expand_tensor(full_sa_feature, -2, num_samples)

        # (bs, num_samples, feature_dim) or (bs, num_samples, feature_dim, feature_dim)
        cond_energy = self.net(sa_feature, delta_feature, self.cond_energy_weights, self.cond_energy_biases, full_sa_feature)

        return energy, cond_energy

    def compute_energy(self, sa_feature, delta_feature, full_sa_feature=None):
        """
        compute the conditional energy from the conditional and total state-action-delta features
        :param sa_feature: (feature_dim, bs, sa_feature_dim) or (feature_dim, feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :param delta_feature: (feature_dim, bs, num_samples, delta_feature_dim), global feature used for prediction,
            notice that bs must be 1D
        :param full_sa_feature: (feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :return: energy: (bs, num_samples, feature_dim) or (bs, num_samples, feature_dim, feature_dim)
        """
        if self.dot_product_energy:
            return self.compute_energy_dot(sa_feature, delta_feature, full_sa_feature)
        else:
            return self.compute_energy_net(sa_feature, delta_feature, full_sa_feature)

    def forward_step(self, actions, obss, rewards, forward_mode=("full", "mask", "causal")):
        """
        # TODO: change these extract features function.
        #   Figure our full energy first.
        #   Replace delta_feature by reward, replace sa features by oa features
        compute energy for the following combinations
        if using (1) next_feature + neg_delta_features for training
            a. feature from randomly masking one state variable + conditional feature from all variables
            b. feature from all variables
            c. feature from causal parents + conditional feature from all variable (? probably for eval only)
        elif using (2) pred_delta_features for evaluation
            a. feature from causal parents + conditional feature from all variable (? probably for eval only)
        :param features:
            if state space is continuous: (bs, num_observation_steps, feature_dim).
            else: NotImplementedError
            notice that bs must be 1D
        :param action: (bs, action_dim)
        :param delta_features:
            if observation space is continuous: (bs, num_samples, feature_dim).
            else: NotImplementedError
        :param forward_mode: which energy to compute
        :return:
        energy
            for training, (bs, 1 + num_negative_samples, feature_dim)
            for evaluation, (bs, num_pred_samples, feature_dim)
        """
        bs, _, feature_dim = rewards.shape
        reward_dim = self.reward_dim
        action_part_dim = self.action_part_dim

        obs_feature = self.extract_observation_feature(obss)
        actions_feature = self.extract_action_part_feature(actions)
        reward_feature = self.extract_reward_feature(rewards)
        ao_feature = torch.cat([actions_feature, obs_feature], dim=1)  # (reward_dim, action_dim + 1, bs, out_dim)

        if self.aggregation == "max":
            full_ao_feature, _ = ao_feature.max(dim=1)                  # (feature_dim, bs, out_dim)
        elif self.aggregation == "mean":
            full_ao_feature = ao_feature.mean(dim=1)                    # (feature_dim, bs, out_dim)
        else:
            raise NotImplementedError

        # (bs, num_samples, feature_dim)
        full_energy = mask_energy = mask_cond_energy = causal_energy = causal_cond_energy = None

        if "full" in forward_mode:
            full_energy = self.compute_energy(full_ao_feature, reward_feature)  # (bs, num_samples, feature_dim)

        if "mask" in forward_mode:
            mask_sa_feature = ao_feature.clone()  # (feature_dim, feature_dim + 1, bs, out_dim)
            if self.train_all_masks or not self.training:
                mask = self.training_masks  # (feature_dim, feature_dim, feature_dim + 1, 1, 1)
                mask_sa_feature = mask_sa_feature.unsqueeze(dim=1)  # (feature_dim, 1, feature_dim + 1, bs, out_dim)
            else:
                mask = self.get_training_mask(bs)  # (bs, feature_dim, feature_dim + 1)
                mask = torch.permute(mask, (1, 2, 0))  # (feature_dim, feature_dim + 1, bs)
                mask = mask.unsqueeze(dim=-1)  # (feature_dim, feature_dim + 1, bs, 1)

            # (feature_dim, feature_dim, feature_dim + 1, bs, out_dim) or (feature_dim, feature_dim + 1, bs, out_dim)
            mask_sa_feature = mask_sa_feature * mask

            # (feature_dim, feature_dim, bs, out_dim) or (feature_dim, bs, out_dim)
            if self.aggregation == "max":
                mask_sa_feature, _ = mask_sa_feature.max(dim=-3)
            elif self.aggregation == "mean":
                raise NotImplementedError
                mask_sa_feature = mask_sa_feature.sum(dim=-3) / feature_dim
            else:
                raise NotImplementedError

            # (bs, num_samples, feature_dim) or (bs, num_samples, feature_dim, feature_dim)
            mask_energy, mask_cond_energy = self.compute_energy(mask_sa_feature, reward_feature, full_ao_feature)

        if "causal" in forward_mode:
            causal_sa_feature = ao_feature.clone()  # (feature_dim, feature_dim + 1, bs, out_dim)
            causal_mask = self.mask.detach().view(reward_dim, action_part_dim + 1, 1, 1)
            causal_sa_feature = causal_sa_feature * causal_mask  # (feature_dim, feature_dim + 1, bs, out_dim)

            if self.aggregation == "max":
                causal_sa_feature, _ = causal_sa_feature.max(dim=1)  # (feature_dim, bs, out_dim)
            elif self.aggregation == "mean":
                num_parents = causal_mask.sum(dim=1)
                causal_sa_feature = causal_sa_feature.sum(dim=1) / num_parents  # (feature_dim, bs, out_dim)
            else:
                raise NotImplementedError

            # (bs, num_samples, feature_dim)
            causal_energy, causal_cond_energy = self.compute_energy(causal_sa_feature, reward_feature,
                                                                    full_ao_feature)

        return full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy


    def forward_with_feature(self, actions, obss, rewards,
                             forward_mode=("full", "mask", "causal")):
        """
        :param actions:
            if observation space is continuous: (bs, num_observation_steps, feature_dim).
            else: NotImplementedError
            notice that bs can be a multi-dimensional batch size
        :param obss: (bs, num_pred_steps, action_dim)
        :param rewards:
            if observation space is continuous: (bs, num_pred_steps, feature_dim).
            else: NotImplementedError
        :param neg_delta_features:
            if observation space is continuous: (bs, num_pred_steps, num_negative_samples, feature_dim).
            else: NotImplementedError
        :param forward_mode: which energy to compute
        :return: energy:
            if observation space is continuous: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
            else: NotImplementedError
        """
        num_observation_steps, _ = actions.shape[-2:]
        bs = actions.shape[:-2]
        if self.parallel_sample and self.training:
            bs = bs[1:]
        reward_dim = self.reward_dim
        action_part_dim = self.action_part_dim

        # This is how we should proceed
        assert (num_observation_steps == 1)

        num_pred_steps = num_observation_steps



        if num_pred_steps > 1:
            raise NotImplementedError

        # This never gets executed?... Ok still need this during inference
        flatten_bs = len(bs) > 1
        if flatten_bs:
            import ipdb
            ipdb.set_trace()

        if self.parallel_sample and self.training:
            obss = obss[:, :, 0, ...]                                         # (bs, ladar, n_channels)
        else:
            obss = obss[:, 0, ...]  # (bs, ladar, n_channels)
        obss.requires_grad = True

        actions.requires_grad = True
        delta_feature = rewards.detach()                                    # (bs, 1, feature_dim)

        delta_feature = delta_feature[..., 0, :]

        if self.parallel_sample and self.training:
            eye = torch.eye(reward_dim, device=self.device).unsqueeze(dim=-2)
            delta_feature = (delta_feature * eye).sum(dim=-1).T

        delta_feature = delta_feature.unsqueeze(dim=-2)                     # (bs, 1, feature_dim)


        # sample negative delta features based on current delta fetaures
        # For now, treat input as array rather than dictionaries (we can fix this later)
        # (bs, num_pred_steps, num_negative_samples, feature_dim)
        neg_delta_features = self.sample_boolean_neg_feature(bs + (num_pred_steps,), delta_feature.detach())
        num_negative_samples = neg_delta_features.shape[-2]

        # TODO: add this if using standard neg sample
        # neg_delta_features = neg_delta_features[:, 0]                       # (bs, num_negative_samples, feature_dim)
        delta_features = torch.cat([delta_feature, neg_delta_features], dim=-2)   # (bs, 1 + num_negative_samples, feature_dim)

        delta_features.requires_grad = True
        grad_tensors = (actions, obss, delta_features)

        full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy = \
            self.forward_step(actions, obss, delta_features, forward_mode)

        # TODO: figure out the exact dimensions

        # (bs, 1, 1 + num_negative_samples, feature_dim)
        if "full" in forward_mode:
            full_energy = full_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, reward_dim)
        if "mask" in forward_mode:
            if mask_energy.ndim == 4:
                mask_energy = mask_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, reward_dim, action_part_dim)
                mask_cond_energy = mask_cond_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, reward_dim, action_part_dim)
            elif mask_energy.ndim == 3:
                mask_energy = mask_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, reward_dim)
                mask_cond_energy = mask_cond_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, reward_dim)
            else:
                raise NotImplementedError
        if "causal" in forward_mode:
            causal_energy = causal_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, reward_dim)
            causal_cond_energy = causal_cond_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, reward_dim)

        return full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy, grad_tensors

    # We assume that args are passed in as tensor instead of dictionary
    def forward(self, actions, obss, rewards, forward_mode=("full", "mask", "causal")):
        # features = self.get_feature(actions)
        # next_features = self.get_feature(rewards)
        return self.forward_with_feature(actions, obss, rewards, forward_mode)

    # modified to fit our scheme
    # We should never mask out the last dimension
    def get_mask_by_id(self, mask_ids):
        """
        :param mask_ids: (bs feature_dim), idxes of state variable to drop
            notice that bs can be a multi-dimensional batch size
        :return: (bs, feature_dim, feature_dim + 1), bool mask of state variables to use
        """
        int_mask = F.one_hot(mask_ids, self.action_part_dim + 1)
        bool_mask = int_mask < 1
        return bool_mask

    def get_training_mask(self, bs):
        # uniformly select one state variable to omit when predicting the next time step value
        if isinstance(bs, int):
            bs = (bs,)

        idxes = torch.randint(self.action_part_dim, bs + (self.reward_dim,), device=self.device)
        return self.get_mask_by_id(idxes)  # (bs, feature_dim, feature_dim + 1)

    # Modified so that matches our desired dimension
    def get_eval_mask(self, bs, i):
        # omit i-th state variable or the action when predicting the next time step value

        if isinstance(bs, int):
            bs = (bs,)

        feature_dim = self.reward_dim
        idxes = torch.full(size=bs + (feature_dim,), fill_value=i, dtype=torch.int64, device=self.device)

        # # We don't need any of these: we simply mask out the given i
        # # this is quite hacky tbh
        # # each state variable must depend on itself when predicting the next time step value
        # self_mask = torch.arange(feature_dim, device=self.device)
        # idxes[idxes >= self_mask] += 1

        return self.get_mask_by_id(idxes)  # (bs, feature_dim, feature_dim + 1)

    def bo_loss(self, energy, cond_energy):
        """
        :param energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim) or 
                       (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
        :param cond_energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim) or 
                            (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
        :return:
            loss: scalar
        """
        return self.nce_loss(energy.detach() + cond_energy)

    @staticmethod
    def energy_norm_loss(energy):
        """
        :param energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim) or 
                       (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
        :return:
            loss: scalar
        """
        if energy.ndim == 4:
            energy_sq = (energy ** 2).sum(dim=(-3, -1)).mean()
            energy_abs = energy.abs().sum(dim=(-3, -1)).mean()
        elif energy.ndim == 5:
            energy_sq = (energy ** 2).sum(dim=(-4, -2, -1)).mean()
            energy_abs = energy.abs().sum(dim=(-4, -2, -1)).mean()
        else:
            raise NotImplementedError

        norm_reg_coef = 1e-6
        return energy_sq * norm_reg_coef, energy_abs

    @staticmethod
    def energy_grad_loss(energy, tensors):
        """
        :param energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim) or 
                       (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
        :param tensors: a list of tensors
        :return:
            gradient: (bs, 1 + num_negative_samples, feature_dim)
        """
        grad_reg_coef = 1e-6
        grad_thre = 0

        if energy.ndim == 4:
            grads = torch.autograd.grad(energy.sum(), tensors, create_graph=True)
        elif energy.ndim == 5:
            feature_dim = energy.shape[-1]
            grads = [torch.autograd.grad(energy[..., i].sum(), tensors, create_graph=True)
                     for i in range(feature_dim)]
            grads = list(map(list, zip(*grads)))
            grads = [torch.stack(grad, dim=-1) for grad in grads]
        else:
            raise NotImplementedError
        grads_abs = 0
        grads_penalty = 0

        for tensor, grad in zip(tensors, grads):
            if tensor.ndim == 2:
                bs = tensor.shape[0]
                grad = grad.view(bs, -1)
            elif tensor.ndim == 3:
                bs, num_samples_or_steps, _ = tensor.shape
                grad = grad.view(bs * num_samples_or_steps, -1)
            else:
                raise NotImplementedError
            grads_abs += grad.abs().mean()
            grads_penalty += (F.relu(grad.abs() - grad_thre) ** 2).sum(dim=-1).mean()

        return grads_penalty * grad_reg_coef, grads_abs

    def update(self, actions, obss, rewards, eval=False):
        """
        :param actions: {obs_i_key: (bs, num_observation_steps, obs_i_shape)}
            notice that bs can be a multi-dimensional batch size
        :param obss: (bs, num_pred_steps, obs_dim, obs_channel)
        :param rewards: ({obs_i_key: (bs, num_pred_steps, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """
        if eval:
            return self.update_mask(actions, obss, rewards)

        self.update_num += 1

        bs, num_pred_steps = actions.shape[:-2], actions.shape[-2]

        if self.parallel_sample:
            bs = bs[1:]

        forward_mode = []
        opt_mask = self.mask_opt_freq > 0 and self.update_num % self.mask_opt_freq == 0
        opt_full = self.full_opt_freq > 0 and self.update_num % self.full_opt_freq == 0
        opt_causal = self.causal_opt_freq > 0 and self.update_num % self.causal_opt_freq == 0
        imit_full = False
        if opt_mask:
            forward_mode.append("mask")
        if opt_full or (opt_mask and imit_full):
            forward_mode.append("full")
        if self.use_prioritized_buffer or opt_causal:
            forward_mode.append("causal")

        # (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy, grad_tensors = \
            self.forward(actions, obss, rewards, forward_mode)
        features, action, delta_features = grad_tensors

        grad_tensors = (delta_features,)

        loss = 0
        loss_detail = {}
        if "mask" in forward_mode:
            # mask_energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim) or 
            #              (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
            mask_nce_loss = self.nce_loss(mask_energy)
            loss_detail["mask_nce_loss"] = mask_nce_loss
            if opt_mask:
                energy_norm_loss, energy_norm = self.energy_norm_loss(mask_energy)
                energy_grad_loss, energy_grad = self.energy_grad_loss(mask_energy, grad_tensors)
                loss += mask_nce_loss + energy_norm_loss + energy_grad_loss
                loss_detail["mask_energy_norm"] = energy_norm
                loss_detail["mask_energy_grad"] = energy_grad

                if self.learn_bo:
                    mask_bo_loss = self.bo_loss(mask_energy, mask_cond_energy)
                    loss += mask_bo_loss
                    loss_detail["mask_bo_gain"] = mask_nce_loss - mask_bo_loss

        if "full" in forward_mode:
            full_nce_loss = self.nce_loss(full_energy)
            loss_detail["full_nce_loss"] = full_nce_loss

            if opt_mask and imit_full:
                mask_cond_energy = full_energy.detach() - mask_energy
                energy_norm_loss, energy_norm = self.energy_norm_loss(mask_cond_energy)
                energy_grad_loss, energy_grad = self.energy_grad_loss(mask_cond_energy, grad_tensors)
                loss += energy_norm_loss + energy_grad_loss

            if opt_full:
                energy_norm_loss, energy_norm = self.energy_norm_loss(full_energy)
                energy_grad_loss, energy_grad = self.energy_grad_loss(full_energy, grad_tensors)
                loss += full_nce_loss + energy_norm_loss + energy_grad_loss
                loss_detail["full_energy_norm"] = energy_norm
                loss_detail["full_energy_grad"] = energy_grad

        if "causal" in forward_mode:
            causal_nce_loss = self.nce_loss(causal_energy)
            loss_detail["causal_nce_loss"] = causal_nce_loss

            if self.use_prioritized_buffer:
                priority = 1 - F.softmax(causal_energy, dim=-2)[..., 0, :].mean(dim=-2)         # (bs, feature_dim)

                if self.parallel_sample:
                    priority = priority.T
                else:
                    priority = priority.mean(dim=-1)
                loss_detail["priority"] = priority

            if opt_causal:
                energy_norm_loss, energy_norm = self.energy_norm_loss(causal_energy)
                energy_grad_loss, energy_grad = self.energy_grad_loss(causal_energy, grad_tensors)
                loss += causal_nce_loss + energy_norm_loss + energy_grad_loss
                loss_detail["causal_energy_norm"] = energy_norm
                loss_detail["causal_energy_grad"] = energy_grad

            if self.learn_bo:
                causal_bo_loss = self.bo_loss(causal_energy, causal_cond_energy)
                loss += causal_bo_loss
                loss_detail["causal_bo_gain"] = causal_nce_loss - causal_bo_loss

        self.backprop(loss, loss_detail)

        return loss_detail

    @staticmethod
    def compute_cmi(energy, cond_energy, unbiased=True):
        """
        https://arxiv.org/pdf/2106.13401, proposition 3
        :param energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
            notice that bs can be a multi-dimensional batch size
        :param cond_energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
        :return: cmi: (feature_dim,feature_dim) (previous documentation is wrong)
        """
        pos_cond_energy = cond_energy[..., 0, :, :]         # (bs, num_pred_steps, feature_dim, feature_dim)

        if unbiased:
            K = energy.shape[-3]                            # num_negative_samples
            neg_energy = energy[..., 1:, :, :]              # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)
            neg_cond_energy = cond_energy[..., 1:, :, :]    # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)
        else:
            K = energy.shape[-3] + 1                        # num_negative_samples
            neg_energy = energy                             # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)
            neg_cond_energy = cond_energy                   # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)

        log_w_neg = F.log_softmax(neg_energy, dim=-3)       # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)
        # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)
        weighted_neg_cond_energy = np.log(K - 1) + log_w_neg + neg_cond_energy
        # (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
        cond_energy = torch.cat([pos_cond_energy.unsqueeze(dim=-3), weighted_neg_cond_energy], dim=-3)
        log_denominator = -np.log(K) + torch.logsumexp(cond_energy, dim=-3)         # (bs, num_pred_steps, feature_dim, feature_dim)
        cmi = pos_cond_energy - log_denominator                                     # (bs, num_pred_steps, feature_dim, feature_dim)

        cmi_dim = cmi.shape[-2:]
        cmi = cmi.sum(dim=-3).view(-1, *cmi_dim).mean(dim=0)
        return cmi

    def update_mask(self, actions, obss, rewards):
        """
        :param actions: {obs_i_key: (bs, num_observation_steps, obs_i_shape)}
            notice that bs can be a multi-dimensional batch size
        :param obss: (bs, num_pred_steps, action_dim)
        :param rewards: ({obs_i_key: (bs, num_pred_steps, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """
        bs, num_pred_steps = actions.shape[:-2], actions.shape[-2]
        # feature_dim = self.feature_dim

        with torch.no_grad():
            # features = self.encoder(actions)
            # next_features = self.encoder(rewards)
            features = actions
            next_features = rewards

            full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy, _ = \
                self.forward_with_feature(features, obss, next_features)

            mask_nce_loss = self.nce_loss(mask_energy)
            mask_bo_loss = self.bo_loss(mask_energy, mask_cond_energy)
            full_nce_loss = self.nce_loss(full_energy)
            causal_nce_loss = self.nce_loss(causal_energy)
            causal_bo_loss = self.bo_loss(causal_energy, causal_cond_energy)

            eval_details = {"mask_nce_loss": mask_nce_loss,
                            "mask_bo_gain": mask_nce_loss - mask_bo_loss,
                            "full_nce_loss": full_nce_loss,
                            "causal_nce_loss": causal_nce_loss,
                            "causal_bo_gain": causal_nce_loss - causal_bo_loss}

            if not self.learn_bo:
                mask_cond_energy = full_energy.unsqueeze(dim=-1) - mask_energy

            # energy: (bs, num_samples, reward_dim)
            cmi = self.compute_cmi(mask_energy, mask_cond_energy)                       # (reward_dim, action_dim)


        reward_dim = self.reward_dim
        action_part_dim = self.action_part_dim

        diag = torch.ones(reward_dim, action_part_dim + 1, dtype=torch.float32, device=self.device)
        diag *= self.CMI_threshold

        # # (feature_dim, feature_dim), (feature_dim, feature_dim)
        # upper_tri, lower_tri = torch.triu(cmi), torch.tril(cmi, diagonal=-1)
        # diag[:, 1:] += upper_tri
        diag[:, :-1] = cmi

        eval_tau = self.cmi_params.eval_tau
        self.mask_CMI = self.mask_CMI * eval_tau + diag * (1 - eval_tau)
        self.mask = self.mask_CMI >= self.CMI_threshold
        self.mask[self.diag_mask] = True

        return eval_details

    def get_mask(self):
        return self.mask

    def get_adjacency(self):
        return self.mask_CMI

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "mask_CMI": self.mask_CMI,
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("contrastive loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.mask_CMI = checkpoint["mask_CMI"]
            self.mask = self.mask_CMI >= self.CMI_threshold
            self.mask_CMI[self.diag_mask] = self.CMI_threshold
            self.mask[self.diag_mask] = True
