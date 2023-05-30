import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions.one_hot_categorical import OneHotCategorical

from model.inference import Inference
from model.gumbel import gumbel_sigmoid
from model.inference_utils import reset_layer, forward_network, forward_network_batch, get_state_abstraction
from utils.utils import to_numpy
import ipdb


class InferenceCMI(Inference):
    def __init__(self, encoder, decoder, params):

        self.action_part_dim = params.action_part_dim
        self.reward_dim = params.reward_dim
        self.domain = params.domain
        self.obs_dim = params.obs_dim
        self.cnn_dim = 1

        # Check if we need extra network other than cnn for obs
        self.fc_obs_net = self.obs_dim > self.cnn_dim
        if self.fc_obs_net:
            # individual dimension of each observation. Zero-pad if not the same
            self.obs_ind_dim = params.obs_ind_dim
        self.continuous_obs = True

        self.cmi_params = params.inference_params.cmi_params
        self.init_graph(params, encoder)
        super(InferenceCMI, self).__init__(encoder, decoder, params)
        self.causal_pred_reward_mean = 0
        self.causal_pred_reward_std = 1
        self.pred_diff_reward_std = 1

        self.causal_opt_freq = self.cmi_params.causal_opt_freq

        self.init_abstraction()
        self.init_cache()

        replay_buffer_params = params.training_params.replay_buffer_params
        self.parallel_sample = replay_buffer_params.prioritized_buffer and replay_buffer_params.parallel_sample

        self.update_num = 0

    def init_model(self):
        params = self.params
        cmi_params = self.cmi_params

        # model params
        continuous_reward = self.continuous_reward
        continuous_action = self.continuous_action


        if not self.continuous_action:
            self.action_feature_inner_dim = params.action_feature_inner_dim
        if not self.continuous_reward:
            self.reward_feature_inner_dim = params.reward_feature_inner_dim

        self.scans_extractor = nn.ModuleList()

        # Defining obs net as a CNN
        # The network for image (lidar) processing is different
        img_out = cmi_params.feature_fc_dims[-1]
        if params.domain == "igibson":
            lidar_shape = [220, 1]
            n_input_channels = lidar_shape[1]  # channel last
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
                fc = nn.Sequential(nn.Linear(n_flatten, img_out), nn.ReLU())
                self.scans_extractor.append(nn.Sequential(cnn, fc).to(params.device))
        elif params.domain == "minigrid":
            # Define image embedding
            img_shape = [3, 7, 7]
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
                self.scans_extractor.append(nn.Sequential(cnn, fc).to(params.device))
        else:
            raise NotImplementedError

        action_part_dim = self.action_part_dim
        reward_dim = self.reward_dim
        ar_dim = action_part_dim * reward_dim

        self.action_part_feature_weights = nn.ParameterList()
        self.action_part_feature_biases = nn.ParameterList()

        # assuming the first obs dim is always the image
        if self.fc_obs_net:
            self.obs_feature_weights = nn.ParameterList()
            self.obs_feature_biases = nn.ParameterList()

        self.generative_weights = nn.ParameterList()
        self.generative_biases = nn.ParameterList()

        # only needed for discrete state space
        self.action_part_feature_1st_layer_weights = nn.ParameterList()
        self.action_part_feature_1st_layer_biases = nn.ParameterList()
        self.generative_last_layer_weights = nn.ParameterList()
        self.generative_last_layer_biases = nn.ParameterList()



        # TODO: handle discrete
        # state feature extractor
        if continuous_action:
            in_dim = params.ind_action_dim * self.num_observation_steps
            fc_dims = cmi_params.feature_fc_dims
        else:
            raise NotImplementedError
            out_dim = cmi_params.feature_fc_dims[0]
            fc_dims = cmi_params.feature_fc_dims[1:]
            for feature_i_dim in self.action_feature_inner_dim:
                in_dim = feature_i_dim
                self.action_part_feature_1st_layer_weights.append(nn.Parameter(torch.zeros(reward_dim, in_dim, out_dim)))
                self.action_part_feature_1st_layer_biases.append(nn.Parameter(torch.zeros(reward_dim, 1, out_dim)))
            in_dim = out_dim

        # Action feature extractor
        for out_dim in fc_dims:
            self.action_part_feature_weights.append(nn.Parameter(torch.zeros(ar_dim, in_dim, out_dim)))
            self.action_part_feature_biases.append(nn.Parameter(torch.zeros(ar_dim, 1, out_dim)))
            in_dim = out_dim

        if self.fc_obs_net:
            or_dim = (self.obs_dim - self.cnn_dim) * reward_dim
            in_dim = self.obs_ind_dim
            for out_dim in fc_dims:
                self.obs_feature_weights.append(nn.Parameter(torch.zeros(or_dim, in_dim, out_dim)))
                self.obs_feature_biases.append(nn.Parameter(torch.zeros(or_dim, 1, out_dim)))
                in_dim = out_dim


        # predictor
        in_dim = cmi_params.feature_fc_dims[-1]
        for out_dim in cmi_params.generative_fc_dims:
            self.generative_weights.append(nn.Parameter(torch.zeros(reward_dim, in_dim, out_dim)))
            self.generative_biases.append(nn.Parameter(torch.zeros(reward_dim, 1, out_dim)))
            in_dim = out_dim

        if continuous_reward:
            self.generative_weights.append(nn.Parameter(torch.zeros(reward_dim, in_dim, 2)))
            self.generative_biases.append(nn.Parameter(torch.zeros(reward_dim, 1, 2)))
        else:
            for feature_i_dim in self.reward_feature_inner_dim:
                final_dim = 2 if feature_i_dim == 1 else feature_i_dim
                self.generative_last_layer_weights.append(nn.Parameter(torch.zeros(1, in_dim, final_dim)))
                self.generative_last_layer_biases.append(nn.Parameter(torch.zeros(1, 1, final_dim)))

    def reset_params(self):

        reward_dim = self.reward_dim
        action_part_dim = self.action_part_dim

        if not self.continuous_action:
            for w, b in zip(self.action_part_feature_1st_layer_weights, self.action_part_feature_1st_layer_biases):
                for i in range(reward_dim):
                    reset_layer(w[i], b[i])

        for w, b in zip(self.action_part_feature_weights, self.action_part_feature_biases):
            for i in range(reward_dim * action_part_dim):
                reset_layer(w[i], b[i])

        if self.fc_obs_net:
            for w, b in zip(self.obs_feature_weights, self.obs_feature_biases):
                for i in range( (self.obs_dim - self.cnn_dim) * reward_dim):
                    reset_layer(w[i], b[i])

        for w, b in zip(self.generative_weights, self.generative_biases):
            for i in range(reward_dim):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_last_layer_weights, self.generative_last_layer_biases):
            reset_layer(w, b)

    def init_graph(self, params, encoder):
        device = params.device
        self.CMI_threshold = self.cmi_params.CMI_threshold
        action_part_dim = self.action_part_dim
        reward_dim = self.reward_dim
        # It is called diag mask but in reality it is simply used to ensure that proper elements get masked out
        self.diag_mask = torch.zeros(reward_dim, action_part_dim + self.obs_dim, dtype=torch.bool, device=device)
        self.diag_mask[:, action_part_dim:] = True
        self.mask_CMI = torch.ones(reward_dim, action_part_dim + self.obs_dim, device=device) * self.CMI_threshold
        self.mask = torch.ones(reward_dim, action_part_dim + self.obs_dim, dtype=torch.bool, device=device)

    def init_abstraction(self):
        self.abstraction_quested = False
        self.abstraction_graph = None
        self.action_children_idxes = None

    def init_cache(self):
        # cache for faster mask updates
        self.use_cache = False
        self.sa_feature_cache = None
        self.obs_feature = None
        self.full_action_feature = None
        self.causal_state_feature = None

        # # TODO: enable this once it is clear
        # feature_dim = self.feature_dim
        # self.feature_diag_mask = torch.eye(feature_dim, dtype=torch.float32, device=self.device)
        # self.feature_diag_mask = self.feature_diag_mask.view(feature_dim, feature_dim, 1, 1)

    # modified from extract_action_feature
    def extract_observation_feature(self, obs):
        """
        Modified: takes in a ladar scan and output an obs feature
        (feature_dim) * bs * obs_dim
        """

        if self.obs_dim == self.cnn_dim == 1:
            return self.extract_cnn_obs_feature(obs)
        elif self.fc_obs_net:
            assert(self.obs_dim == 2)
            cnn_feat = self.extract_cnn_obs_feature(obs[0])
            fc_feat = self.extact_fc_obs_feature(obs[1])
            obs_features = torch.cat((cnn_feat, fc_feat), dim=1)
        else:
            raise NotImplementedError

        return obs_features

    def extract_cnn_obs_feature(self, obs):
        """
        Extract the observation features that rely on CNN (These are written manuelly for now)
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
            obs_features.append(self.scans_extractor[i](obs[i]))

        obs_features = torch.stack(obs_features)
        dim_out = obs_features.shape[-1]
        obs_features = obs_features.reshape([self.reward_dim, 1, -1, dim_out])

        return obs_features


    def extact_fc_obs_feature(self, obs):
        """
        :param obs:
            if state space is continuous: (bs, num_observation_steps, feature_dim).
            else: [(bs, num_observation_steps, feature_i_dim)] * feature_dim
            notice that bs must be 1D
        :return:
            The extracted observation features for things that should be connected using fc layer
        """

        reward_dim = self.reward_dim
        obs_ind_dim = self.obs_ind_dim
        obs_fc_dim = self.obs_dim - self.cnn_dim

        assert(obs_fc_dim == 1)

        # # Ignore discrete obs
        assert(self.continuous_obs == True)
        if self.parallel_sample and self.training:
            raise NotImplementedError
            bs = obs.shape[1]
            x = obs.permute(0, 3, 1, 2)  # (feature_dim, feature_dim, bs, num_observation_steps)
        else:
            bs = obs.shape[0]
            x = obs.reshape((bs, obs_ind_dim, obs_fc_dim)).permute(2, 0, 1).unsqueeze(dim=0)  # (1, obs_fc_dim, bs, obs_ind_dim)
            x = x.repeat(reward_dim, 1, 1, 1)  # (reward_dim, action_part_dim, bs, num_observation_steps)
        x = x.reshape(obs_fc_dim * reward_dim, bs, obs_ind_dim)  # (feature_dim * feature_dim, bs, 6)

        # Explain: last dimension is the dim_in (in our case it should not be 1, but 6)
        # Second dim is the dimension of observations, in this case should be 1
        obs_fc_features = forward_network(x, self.obs_feature_weights, self.obs_feature_biases)
        obs_fc_features = obs_fc_features.view(reward_dim, obs_fc_dim, bs, -1)
        return obs_fc_features                                              # (feature_dim, feature_dim, bs, out_dim)

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

        if self.continuous_action:
            if self.parallel_sample and self.training:
                bs = actions.shape[1]
                x = actions.permute(0, 3, 1, 2)  # (feature_dim, feature_dim, bs, num_observation_steps)
            else:
                bs = actions.shape[0]
                x = actions.permute(2, 0, 1).unsqueeze(dim=0)  # (1, action_part_dim, bs, num_observation_steps)
                x = x.repeat(reward_dim, 1, 1, 1)  # (reward_dim, action_part_dim, bs, num_observation_steps)
            x = x.reshape(action_part_dim * reward_dim, bs, -1)  # (feature_dim * feature_dim, bs, 1)
        else:
            bs = actions[0].shape[0]
            # [(bs, num_observation_steps, feature_i_dim)] * feature_dim
            reshaped_feature = []
            for f_i in actions:
                f_i = f_i.repeat(reward_dim, 1, 1)  # (feature_dim, bs, feature_i_dim)
                reshaped_feature.append(f_i)

            import ipdb
            ipdb.set_trace()

            x = forward_network_batch(reshaped_feature,
                                      self.action_part_feature_1st_layer_weights,
                                      self.action_part_feature_1st_layer_biases)
            x = torch.stack(x, dim=1)  # (feature_dim, feature_dim, bs, out_dim)
            x = x.view(action_part_dim * reward_dim, *x.shape[2:])  # (feature_dim * feature_dim, bs, out_dim)

        actions_feature = forward_network(x, self.action_part_feature_weights, self.action_part_feature_biases)
        actions_feature = actions_feature.view(reward_dim, action_part_dim, bs, -1)
        return actions_feature                                              # (feature_dim, feature_dim, bs, out_dim)

    def extract_masked_action_feature(self, masked_feature, full_action_feature):
        """
        :param masked_feature:
            if state space is continuous: (bs, num_observation_steps, feature_dim).
            else: [(bs, num_observation_steps, feature_i_dim)] * feature_dim
            notice that bs must be 1D
        :param full_action_feature: (reward_dim, action_part_dim, bs, out_dim), calculated by self.extract_state_feature()
        :return: (feature_dim, feature_dim, bs, out_dim),
            the first feature_dim is each state variable at next time step to predict, the second feature_dim are
            inputs (all current state variables) for the prediction
        """
        raise NotImplementedError

        action_part_dim = self.action_part_dim
        reward_dim = self.reward_dim

        if self.continuous_action:
            x = masked_feature.permute(2, 0, 1)                             # (feature_dim, bs, num_observation_steps)
        else:
            # [(1, bs, feature_i_dim)] * feature_dim
            masked_feature = [f_i.unsqueeze(dim=0) for f_i in masked_feature]
            x = forward_network_batch(masked_feature,
                                      [w[i:i+1] for i, w in enumerate(self.action_part_feature_1st_layer_weights)],
                                      [b[i:i+1] for i, b in enumerate(self.action_part_feature_1st_layer_biases)])
            x = torch.cat(x, dim=0)                                         # (feature_dim, bs, out_dim)


        # Notice: Originally it would return the diagonal, which is not what we want.
        #  Right now we don't even call this function
        #  Figure out what this does: w(ar_dim: reward * action, in_dim, out_dim)))

        idxes = [i * (action_part_dim + 1) for i in range(reward_dim)]
        x = forward_network(x,
                            [w[idxes] for w in self.action_part_feature_weights],
                            [b[idxes] for b in self.action_part_feature_biases])  # (feature_dim, bs, out_dim)


        # feature_diag_mask = self.feature_diag_mask                          # (feature_dim, feature_dim, 1, 1)
        masked_action_feature = x.unsqueeze(dim=0)                           # (1, feature_dim, bs, out_dim)
        # masked_action_feature = full_action_feature * (1 - feature_diag_mask) + masked_action_feature * feature_diag_mask

        return masked_action_feature                                         # (feature_dim, feature_dim, bs, out_dim)

    def predict_from_sa_feature(self, sa_feature, residual_base=None, abstraction_mode=False):
        """
        predict the distribution and sample for the next step value of all state variables
        :param sa_feature: (feature_dim, bs, sa_feature_dim), global feature used for prediction,
            notice that bs can be a multi-dimensional batch size
        :param residual_base: (bs, feature_dim), residual used for continuous state variable prediction
        :param abstraction_mode: if the prediction is computed for state variables in the abstraction only.
            If True, all feature_dim in this function should be replaced by abstraction_feature_dim when indicating
            shapes of tensors.
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * feature_dim, each of shape (bs, feature_i_dim)
        """
        if abstraction_mode:
            generative_weights = self.abstraction_generative_weights
            generative_biases = self.abstraction_generative_biases
            generative_last_layer_weights = self.abstraction_generative_last_layer_weights
            generative_last_layer_biases = self.abstraction_generative_last_layer_biases
        else:
            generative_weights, generative_biases = self.generative_weights, self.generative_biases
            generative_last_layer_weights = self.generative_last_layer_weights
            generative_last_layer_biases = self.generative_last_layer_biases

        x = forward_network(sa_feature, generative_weights, generative_biases)
        def normal_helper(mean_, base_, log_std_):
            if self.residual:
                mean_ = mean_ + base_
            log_std_ = torch.clip(log_std_, min=self.log_std_min, max=self.log_std_max)
            std_ = torch.exp(log_std_)
            return Normal(mean_, std_)

        if self.continuous_reward:
            x = x.permute(1, 0, 2)                                          # (bs, feature_dim, 2)
            mu, log_std = x.unbind(dim=-1)                                  # (bs, feature_dim) * 2
            return normal_helper(mu, residual_base, log_std)
        else:
            x = F.relu(x)                                                   # (feature_dim, bs, out_dim)
            x = [x_i.unsqueeze(dim=0) for x_i in torch.unbind(x, dim=0)]    # [(1, bs, out_dim)] * feature_dim
            x = forward_network_batch(x,
                                      generative_last_layer_weights,
                                      generative_last_layer_biases,
                                      activation=None)

            feature_inner_dim = self.reward_feature_inner_dim
            if abstraction_mode:
                raise NotImplementedError
                feature_inner_dim = feature_inner_dim

            dist = []
            for base_i, feature_i_inner_dim, dist_i in zip(residual_base, feature_inner_dim, x):
                dist_i = dist_i.squeeze(dim=0)
                if feature_i_inner_dim == 1:
                    mu, log_std = torch.split(dist_i, 1, dim=-1)            # (bs, 1), (bs, 1)
                    dist.append(normal_helper(mu, base_i, log_std))
                else:
                    dist.append(OneHotCategorical(logits=dist_i))
            return dist

    def forward_step(self, full_feature, masked_feature, causal_feature, obs, mask=None,
                     obs_feature=None, full_action_feature=None):
        """
        :param full_feature: if state space is continuous: (bs, feature_dim).
            Otherwise: [(bs, feature_i_dim)] * feature_dim
            if it is None, no need to forward it
        :param masked_feature: (bs, feature_dim) or [(bs, feature_i_dim)] * feature_dim
        :param causal_feature: (bs, feature_dim) or [(bs, feature_i_dim)] * feature_dim
        :param obs: (bs, action_dim)
        :param mask: (bs, feature_dim, feature_dim + 1)
        :param obs_feature: (bs, feature_dim, 1, out_dim), pre-cached value
        :param full_action_feature: (bs, feature_dim, feature_dim, out_dim), pre-cached value
        :param no_causal: not to forward causal_feature, used for training
        :param causal_only: whether to only forward causal_feature, used for curiosity reward and model-based roll-out
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * feature_dim, each of shape (bs, feature_i_dim)
        """
        forward_full = full_feature is not None
        forward_masked = masked_feature is not None
        forward_causal = causal_feature is not None

        full_dist = masked_dist = causal_dist = None

        if obs_feature is None:
            # extract features of the action
            # (feature_dim, 1, bs, out_dim)
            self.obs_feature = obs_feature = self.extract_observation_feature(obs)

        if forward_full:
            # 1. extract features of all state variables
            if full_action_feature is None:
                # (feature_dim, feature_dim, bs, out_dim)
                self.full_action_feature = full_action_feature = self.extract_action_part_feature(full_feature)

            # 2. extract global feature by element-wise max
            # (feature_dim, feature_dim + 1, bs, out_dim)
            full_sa_feature = torch.cat([full_action_feature, obs_feature], dim=1)
            full_sa_feature, full_sa_indices = full_sa_feature.max(dim=1)           # (feature_dim, bs, out_dim)

            # 3. predict the distribution of next time step value
            full_dist = self.predict_from_sa_feature(full_sa_feature, full_feature[..., -1, :])

        if forward_masked:
            # 1. extract features of all state variables and the action
            # (feature_dim, feature_dim, bs, out_dim)
            masked_action_feature = self.extract_action_part_feature(masked_feature)
            # TODO: figure out what this does and perhaps add this back
            # self.extract_masked_action_feature(masked_feature, full_action_feature)

            # 2. extract global feature by element-wise max
            # mask out unused features
            # (feature_dim, feature_dim + 1, bs, out_dim)
            masked_sa_feature = torch.cat([masked_action_feature, obs_feature], dim=1)
            mask = mask.permute(1, 2, 0)                                            # (feature_dim, feature_dim + 1, bs)
            masked_sa_feature[~mask] = float('-inf')
            masked_sa_feature, masked_sa_indices = masked_sa_feature.max(dim=1)     # (feature_dim, bs, out_dim)

            # 3. predict the distribution of next time step value
            masked_dist = self.predict_from_sa_feature(masked_sa_feature, masked_feature[..., -1, :])

        if forward_causal:
            # 1. extract features of all state variables and the action
            causal_action_feature = self.extract_action_part_feature(causal_feature)

            # 2. extract global feature by element-wise max
            # mask out unused features
            # (feature_dim, feature_dim + 1, bs, out_dim)
            causal_sa_feature = torch.cat([causal_action_feature, obs_feature], dim=1)
            eval_mask = self.mask.detach()                                          # (feature_dim, feature_dim + 1)
            causal_sa_feature[~eval_mask] = float('-inf')
            causal_sa_feature, causal_sa_indices = causal_sa_feature.max(dim=1)     # (feature_dim, bs, out_dim)

            # 3. predict the distribution of next time step value
            causal_dist = self.predict_from_sa_feature(causal_sa_feature, causal_feature[..., -1, :])

        return full_dist, masked_dist, causal_dist

    def extract_action_feature_abstraction(self, action):
        """
        :param action: (bs, action_dim). notice that bs must be 1D
        :return: {action_children_idx: (1, bs, out_dim)}
        """
        num_action_children = len(self.action_children_idxes)
        action = action.unsqueeze(dim=0)                                    # (1, bs, action_dim)
        action = action.expand(num_action_children, -1, -1)                 # (num_action_children, bs, action_dim)
        # (num_action_children, bs, out_dim)
        action_feature = forward_network(action,
                                         self.abstraction_action_feature_weights,
                                         self.abstraction_action_feature_biases)
        action_feature = action_feature.unsqueeze(dim=1)                    # (num_action_children, 1, bs, out_dim)
        action_feature = torch.unbind(action_feature, dim=0)                # [(1, bs, out_dim)] * num_action_children
        action_feature_dict = {idx: action_feature_i
                               for idx, action_feature_i in zip(self.action_children_idxes, action_feature)}

        return action_feature_dict                                          # {action_children_idx: (1, bs, out_dim)}

    def extract_state_feature_abstraction(self, feature):
        """
        :param feature:
            if state space is continuous: (bs, abstraction_feature_dim).
            else: [(bs, feature_i_dim)] * abstraction_feature_dim
            notice that bs must be 1D
        :return: {state_variable_idx: (num_parent, bs, out_dim)}
        """
        raise NotImplementedError
        if self.continuous_action:
            feature = feature.transpose(0, 1)                                   # (abstraction_feature_dim, bs)

        features = []
        for idx, parent_idxes in self.abstraction_adjacency.items():
            feature_idx = [self.abstraction_idxes.index(parent_idx) for parent_idx in parent_idxes]
            if self.continuous_action:
                x = feature[feature_idx]                                        # (num_parent, bs)
                x = x.unsqueeze(dim=-1)                                         # (num_parent, bs, 1)
                features.append(x)
            else:
                x = [feature[parent_idx] for parent_idx in feature_idx]         # [(bs, feature_i_dim)] * num_parent
                x = [x_i.unsqueeze(dim=0) for x_i in x]                         # [(1, bs, feature_i_dim)] * num_parent
                state_feature_1st_layer_weights = self.abstraction_state_feature_1st_layer_weights[idx]
                state_feature_1st_layer_biases = self.abstraction_state_feature_1st_layer_biases[idx]
                x = forward_network_batch(x,
                                          state_feature_1st_layer_weights,
                                          state_feature_1st_layer_biases)       # [(1, bs, out_dim)] * num_parent
                features.extend(x)
        features = torch.cat(features, dim=0)                                   # (total_num_parent, bs, 1)

        state_feature = forward_network(features,
                                        self.abstraction_state_feature_weights,
                                        self.abstraction_state_feature_biases)

        state_feature_dict = {}
        offset = 0
        for idx, parent_idxes in self.abstraction_adjacency.items():
            num_parents = len(parent_idxes)
            state_feature_dict[idx] = state_feature[offset:offset + num_parents]    # (num_parent, bs, out_dim)
            offset += num_parents
        return state_feature_dict

    def forward_step_abstraction(self, abstraction_feature, action):
        """
        :param abstraction_feature: if state space is continuous: (bs, abstraction_feature_dim)
            Otherwise: [(bs, feature_i_dim)] * abstraction_feature_dim
        :param action: (bs, action_dim)
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous: a Normal distribution of shape (bs, abstraction_feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * abstraction_feature_dim,
                each of shape (bs, feature_i_dim)
        """

        # 1. extract features of all state variables and the action
        # {action_children_idx: (1, bs, out_dim)}
        action_feature = self.extract_action_feature_abstraction(action)
        # {state_variable_idx: (num_parent, bs, out_dim)}
        state_feature = self.extract_state_feature_abstraction(abstraction_feature)

        # 2. extract global feature by element-wise max
        sa_feature = []
        for idx in self.abstraction_idxes:
            sa_feature_i = state_feature[idx]
            if idx in action_feature:
                action_feature_i = action_feature[idx]                              # (1, bs, out_dim)
                sa_feature_i = torch.cat([sa_feature_i, action_feature_i], dim=0)   # (num_parent + 1, bs, out_dim)
            sa_feature_i, _ = sa_feature_i.max(dim=0)                               # (bs, out_dim)}
            sa_feature.append(sa_feature_i)
        # (abstraction_feature_dim, bs, out_dim)
        sa_feature = torch.stack(sa_feature, dim=0)

        # 3. predict the distribution of next time step value
        dist = self.predict_from_sa_feature(sa_feature, abstraction_feature, abstraction_mode=True)

        return dist

    def forward_with_feature(self, action_features, obss, mask=None,
                             forward_mode=("full", "masked", "causal"), abstraction_mode=False):
        """

        :param action_features:
            if action space is continuous: (bs, num_observation_steps, feature_dim).
            else: [(bs, num_observation_steps, feature_i_dim)] * feature_dim
            notice that bs can be a multi-dimensional batch size
        :param obss: (bs, num_pred_steps, action_dim) if self.continuous_action else (bs, num_pred_steps, 1)
            notice that bs can be a multi-dimensional batch size
        :param mask: (bs, feature_dim, feature_dim + 1),
            randomly generated training mask used when forwarding masked_feature
            notice that bs can be a multi-dimensional batch size
        :param forward_mode
        :param abstraction_mode: whether to only forward controllable & action-relevant state variables,
            used for model-based roll-out
        :return: a single distribution or a list of distributions depending on forward_mode,
            each distribution is of shape (bs, num_pred_steps, feature_dim)
            notice that bs can be a multi-dimensional batch size
        """

        # convert features, actions, mask to 2D tensor if bs is multi-dimensional
        reshaped = False
        bs = action_features.shape[:-2]

        reward_dim = self.reward_dim
        action_part_dim = self.action_part_dim

        if len(bs) > 1:
            raise NotImplementedError
            reshaped = True
            if self.continuous_action:
                action_features = action_features.view(-1, self.action_part_dim)
            else:
                action_features = [feature_i.view(-1, self.action_part_dim) for feature_i in action_features]

            obss = obss.view(-1, *obss.shape[-2:])
            if mask is not None:
                mask = mask.view(-1, *mask.shape[-2:])

        # full_features: prediction using all state variables
        # masked_features: prediction using state variables specified by mask
        # causal_features: prediction using causal parents (inferred so far)
        full_features = action_features if "full" in forward_mode else None
        masked_features = action_features if "masked" in forward_mode else None
        causal_features = action_features if "causal" in forward_mode else None

        if abstraction_mode:
            assert not self.use_cache
            forward_mode = ("causal",)
            full_features = masked_features = None
            if self.abstraction_quested:
                if self.continuous_action:
                    causal_features = causal_features[:, self.abstraction_idxes]
                else:
                    causal_features = [causal_features[idx] for idx in self.abstraction_idxes]

        modes = ["full", "masked", "causal"]
        assert all([ele in modes for ele in forward_mode])
        if "masked" in forward_mode:
            assert mask is not None

        full_dists, masked_dists, causal_dists = [], [], []
        sa_feature_cache = []


        # TODO: We assume only 1 timestep, therefore the obss can be directly passed into the program
        #   Furthermore, cache should always be the first element
        if self.use_cache and self.sa_feature_cache:
            # only used when evaluate with the same state and action a lot in self.update_mask()
            obs_feature, full_action_feature = self.sa_feature_cache[0]
        else:
            obs_feature, full_action_feature = None, None

        if self.fc_obs_net:
            obss = [obs.squeeze(dim=len(bs)) for obs in obss]
        else:
            obss = obss.squeeze(dim=len(bs))

        full_dist, masked_dist, causal_dist = \
            self.forward_step(full_features, masked_features, causal_features, obss, mask,
                              obs_feature, full_action_feature)

        full_dists.append(full_dist)
        masked_dists.append(masked_dist)
        causal_dists.append(causal_dist)

        sa_feature_cache.append((self.obs_feature, self.full_action_feature))

        # Todo: include this when doing multiple timestep
        # # We assume only 1 timestep, therefore the obss can be directly passed into the program
        #
        # obss = torch.unbind(obss, dim=len(bs))                                     # [(bs, action_dim)] * num_pred_steps
        # for i, obs in enumerate(obss):
        #     if self.use_cache and self.sa_feature_cache:
        #         # only used when evaluate with the same state and action a lot in self.update_mask()
        #         obs_feature, full_action_feature = self.sa_feature_cache[i]
        #     else:
        #         obs_feature, full_action_feature = None, None
        #
        #     full_dist = masked_dist = None
        #     if abstraction_mode and self.abstraction_quested:
        #         raise NotImplementedError
        #         causal_dist = self.forward_step_abstraction(causal_features, obs)
        #     else:
        #         full_dist, masked_dist, causal_dist = \
        #             self.forward_step(full_features, masked_features, causal_features, obs, mask,
        #                               obs_feature, full_action_feature)
        #
        #     assert(i == 0)
        #     # next_full_feature = self.sample_from_distribution(full_dist)
        #     # next_masked_feature = self.sample_from_distribution(masked_dist)
        #     # next_causal_feature = self.sample_from_distribution(causal_dist)
        #     # full_features = self.cat_features(full_features, next_full_feature)
        #     # masked_features = self.cat_features(masked_features, next_masked_feature)
        #     # causal_features = self.cat_features(causal_features, next_causal_feature)
        #
        #     full_dists.append(full_dist)
        #     masked_dists.append(masked_dist)
        #     causal_dists.append(causal_dist)
        #
        #     sa_feature_cache.append((self.obs_feature, self.full_action_feature))

        if self.use_cache and self.sa_feature_cache is None:
            self.sa_feature_cache = sa_feature_cache

        dists = [full_dists, masked_dists, causal_dists]
        result_dists = []
        for mode in forward_mode:
            dist = dists[modes.index(mode)]
            dist = self.stack_dist(dist)
            if reshaped:
                dist = self.restore_batch_size_shape(dist, bs)
            result_dists.append(dist)

        if len(forward_mode) == 1:
            return result_dists[0]

        return result_dists

    def restore_batch_size_shape(self, dist, bs):
        # restore multi-dimensional batch size
        if self.continuous_action:
            mu, std = dist.mean, dist.stddev                                    # (bs, num_pred_steps, feature_dim)
            mu = mu.view(*bs, *mu.shape[-2:])                                   # (*bs, num_pred_steps, feature_dim)
            std = std.view(*bs, *std.shape[-2:])                                # (*bs, num_pred_steps, feature_dim)
            return Normal(mu, std)
        else:
            # [(bs, num_pred_steps, feature_i_dim)] * feature_dim
            dist_list = []
            for dist_i in dist:
                if isinstance(dist_i, Normal):
                    mu, std = dist.mean, dist.stddev                            # (bs, num_pred_steps, feature_i_dim)
                    mu = mu.view(*bs, *mu.shape[-2:])                           # (*bs, num_pred_steps, feature_i_dim)
                    std = std.view(*bs, *std.shape[-2:])                        # (*bs, num_pred_steps, feature_i_dim)
                    dist_i = Normal(mu, std)
                elif isinstance(dist_i, OneHotCategorical):
                    logits = dist_i.logits                                      # (bs, num_pred_steps, feature_i_dim)
                    logits = logits.view(*bs, *logits.shape[-2:])               # (*bs, num_pred_steps, feature_i_dim)
                    dist_i = OneHotCategorical(logits=logits)
                else:
                    raise NotImplementedError
                dist_list.append(dist_i)

            return dist_list

    def forward(self, obs, actions, mask=None, forward_mode=("full", "masked", "causal"),
                abstraction_mode=False):
        feature = self.get_feature(obs)
        return self.forward_with_feature(feature, actions, mask, forward_mode, abstraction_mode)

    def setup_annealing(self, step):
        super(InferenceCMI, self).setup_annealing(step)

    # modified to fit our scheme
    # We should never mask out the obs dimensions
    def get_mask_by_id(self, mask_ids):
        """
        :param mask_ids: (bs feature_dim), idxes of state variable to drop
            notice that bs can be a multi-dimensional batch size
        :return: (bs, feature_dim, feature_dim + 1), bool mask of state variables to use
        """
        int_mask = F.one_hot(mask_ids, self.action_part_dim + self.obs_dim)
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

    def prediction_loss_from_multi_dist(self, pred_next_dist, next_feature):
        """
        calculate prediction loss for each prediction distributions
        if use CNN encoder: prediction loss = KL divergence
        else: prediction loss = -log_prob
        :param pred_next_dist:
            a list of prediction distributions under different prediction mode,
            where each element is the next step value for all state variables in the format of distribution as follows,
            if state space is continuous:
                a Normal distribution of shape (bs, num_pred_steps, feature_dim)
            else:
                a list of distributions, [OneHotCategorical / Normal] * feature_dim,
                each of shape (bs, num_pred_steps, feature_i_dim)
        :param next_feature:
            if use a CNN encoder:
                a Normal distribution of shape (bs, num_pred_steps, feature_dim)
            elif state space is continuous:
                a tensor of shape (bs, num_pred_steps, feature_dim)
            else:
                a list of tensors, [(bs, num_pred_steps, feature_i_dim)] * feature_dim
        :return: prediction loss and {"loss_name": loss_value}
        """
        # (bs, num_pred_steps, feature_dim)
        pred_losses = [self.prediction_loss_from_dist(pred_next_dist_i, next_feature)
                       for pred_next_dist_i in pred_next_dist]

        if len(pred_losses) == 2:
            pred_losses.append(None)
        assert len(pred_losses) == 3
        full_pred_loss, masked_pred_loss, causal_pred_loss = pred_losses

        full_pred_loss = full_pred_loss.sum(dim=-1).mean()
        masked_pred_loss = masked_pred_loss.sum(dim=-1).mean()

        pred_loss = full_pred_loss + masked_pred_loss

        pred_loss_detail = {"full_pred_loss": full_pred_loss,
                            "masked_pred_loss": masked_pred_loss}

        if causal_pred_loss is not None:
            causal_pred_loss = causal_pred_loss.sum(dim=-1).mean()
            pred_loss += causal_pred_loss
            pred_loss_detail["causal_pred_loss"] = causal_pred_loss

        return pred_loss, pred_loss_detail

    def update(self, actions, obss, rewards, eval=False):
        """
        :param actions: {obs_i_key: (bs, obs_i_shape)}
        :param obss: (bs, num_pred_steps, action_dim)
        :param rewards: {obs_i_key: (bs, num_pred_steps, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """
        if eval:
            return self.update_mask(actions, obss, rewards)

        num_inference_opt_steps = self.params.training_params.num_inference_opt_steps
        forward_mode = ("full", "masked", "causal")

        bs = actions.size(0)
        mask = self.get_training_mask(bs)                           # (bs, feature_dim, feature_dim + 1)

        feature = actions # self.encoder(actions)
        next_feature = rewards # self.encoder(rewards)
        pred_next_dist = self.forward_with_feature(feature, obss, mask, forward_mode=forward_mode)

        # prediction loss in the state / latent space, (bs, num_pred_steps)
        if self.update_num % self.causal_opt_freq:
            pred_next_dist = pred_next_dist[:2]
        pred_loss, loss_detail = self.prediction_loss_from_multi_dist(pred_next_dist, next_feature)

        loss = pred_loss

        if self.decoder is not None:
            assert isinstance(feature, Distribution)
            recon_loss, recon_loss_detail = \
                self.reconstruction_loss(feature, next_feature, actions, rewards, pred_causal_mask)
            loss = loss + recon_loss
            loss_detail.update(recon_loss_detail)

        self.backprop(loss, loss_detail)
        self.update_num += 1

        return loss_detail

    def update_mask(self, actions, obss, rewards):
        bs = actions.size(0)
        # feature_dim = self.feature_dim
        reward_dim = self.reward_dim
        action_part_dim = self.action_part_dim

        # set up cache for faster computation
        self.use_cache = True
        self.sa_feature_cache = None

        eval_details = {}

        masked_pred_losses = []
        with torch.no_grad():
            # feature = self.encoder(actions)
            # next_feature = self.encoder(rewards)
            feature = actions
            next_feature = rewards

            for i in range(action_part_dim):
                mask = self.get_eval_mask(bs, i)
                if i == 0:
                    pred_next_dists = self.forward_with_feature(feature, obss, mask)
                    # pred_loss: (bs, num_pred_steps, feature_dim)
                    full_pred_loss, masked_pred_loss, eval_pred_loss = \
                        [self.prediction_loss_from_dist(pred_next_dist_i, next_feature, keep_variable_dim=True)
                         for pred_next_dist_i in pred_next_dists]
                else:
                    pred_next_dist = self.forward_with_feature(feature, obss, mask, forward_mode=("masked",))
                    # pred_loss: (bs, num_pred_steps, feature_dim)
                    masked_pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_feature,
                                                                      keep_variable_dim=True)

                masked_pred_loss = masked_pred_loss.mean(dim=1)                         # (bs, feature_dim)
                masked_pred_losses.append(masked_pred_loss)
            full_pred_loss = full_pred_loss.mean(dim=1)[..., None]                      # (bs, feature_dim, 1)
            eval_pred_loss = eval_pred_loss.sum(dim=(1, 2)).mean()                      # scalar
            eval_details["eval_pred_loss"] = eval_pred_loss

        masked_pred_losses = torch.stack(masked_pred_losses, dim=-1)                    # (bs, feature_dim, feature_dim)

        # clean cache
        self.use_cache = False
        self.sa_feature_cache = None
        self.obs_feature = None
        self.full_action_feature = None

        # full_pred_loss uses all state variables + action,
        # while along dim 1 of, masked_pred_losses drops either one state variable or the action
        cmi = masked_pred_losses - full_pred_loss                                       # (bs, feature_dim, feature_dim)
        cmi = cmi.mean(dim=0)                                                           # (feature_dim, feature_dim)

        diag = torch.ones(reward_dim, action_part_dim + self.obs_dim, dtype=torch.float32, device=self.device)
        diag *= self.CMI_threshold
        diag[:, :-self.obs_dim] = cmi

        eval_tau = self.cmi_params.eval_tau
        self.mask_CMI = self.mask_CMI * eval_tau + diag * (1 - eval_tau)
        self.mask = self.mask_CMI >= self.CMI_threshold
        self.mask[self.diag_mask] = True

        return eval_details

    def get_mask(self):
        return self.mask

    def get_state_abstraction(self):
        self.abstraction_quested = True
        abstraction_graph = self.update_abstraction()
        self.update_abstracted_dynamics()
        return abstraction_graph

    def update_abstraction(self):
        raise NotImplementedError
        self.abstraction_graph = get_state_abstraction(to_numpy(self.get_mask()))
        self.abstraction_idxes = list(self.abstraction_graph.keys())

        action_idx = self.feature_dim
        self.action_children_idxes = [idx for idx, parent_idxes in self.abstraction_graph.items()
                                      if action_idx in parent_idxes]
        self.abstraction_adjacency = {}
        for idx, parents in self.abstraction_graph.items():
            self.abstraction_adjacency[idx] = [parent for parent in parents if parent < action_idx]

        return self.abstraction_graph

    def update_abstracted_dynamics(self,):
        raise NotImplementedError
        # only need to calculate action feature for state variables that are children of the action
        action_children_idxes = self.action_children_idxes
        self.abstraction_action_feature_weights = [w[action_children_idxes]
                                                   for w in self.action_feature_weights]
        self.abstraction_action_feature_biases = [b[action_children_idxes]
                                                  for b in self.action_feature_biases]

        # when predicting each state variables in the abstraction, only need to compute state feature for their parents
        feature_dim = self.feature_dim
        self.abstraction_state_feature_1st_layer_weights = {}
        self.abstraction_state_feature_1st_layer_biases = {}
        idxes = []
        for idx, parent_idxes in self.abstraction_adjacency.items():
            idxes.extend([parent_idx + idx * feature_dim for parent_idx in parent_idxes])
            self.abstraction_state_feature_1st_layer_weights[idx] = \
                [w[idx:idx + 1] for i, w in enumerate(self.action_part_feature_1st_layer_weights) if i in parent_idxes]
            self.abstraction_state_feature_1st_layer_biases[idx] = \
                [b[idx:idx + 1] for i, b in enumerate(self.action_part_feature_1st_layer_biases) if i in parent_idxes]

        self.abstraction_state_feature_weights = [w[idxes] for w in self.state_feature_weights]
        self.abstraction_state_feature_biases = [b[idxes] for b in self.state_feature_biases]

        abstraction_idxes = self.abstraction_idxes
        self.abstraction_generative_weights = [w[abstraction_idxes] for w in self.generative_weights]
        self.abstraction_generative_biases = [b[abstraction_idxes] for b in self.generative_biases]
        self.abstraction_generative_last_layer_weights = \
            [w for i, w in enumerate(self.generative_last_layer_weights) if i in abstraction_idxes]
        self.abstraction_generative_last_layer_biases = \
            [b for i, b in enumerate(self.generative_last_layer_biases) if i in abstraction_idxes]

    def get_adjacency(self):
        return self.mask_CMI

    # def get_intervention_mask(self):
    #     return self.mask_CMI[:, -1:]

    def train(self, training=True):
        self.training = training

    def eval(self):
        self.train(training=False)

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "mask_CMI": self.mask_CMI,
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("inference loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.mask_CMI = checkpoint["mask_CMI"]
            self.mask = self.mask_CMI >= self.CMI_threshold
            self.mask_CMI[self.diag_mask] = self.CMI_threshold
            self.mask[self.diag_mask] = True
