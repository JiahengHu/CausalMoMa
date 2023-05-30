import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions.categorical import Categorical

from utils.utils import to_numpy


class Contrastive(nn.Module):
    def __init__(self, encoder, decoder, params):
        super(Contrastive, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.params = params
        self.device = device = params.device
        self.contrastive_params = contrastive_params = params.contrastive_params

        self.continuous_state = params.continuous_action
        self.continuous_action = params.continuous_action
        self.num_observation_steps = params.training_params.num_observation_steps
        self.use_prioritized_buffer = params.training_params.replay_buffer_params.prioritized_buffer

        self.loss_type = contrastive_params.loss_type
        self.l2_reg_coef = contrastive_params.l2_reg_coef
        self.num_pred_steps = contrastive_params.num_pred_steps
        self.gradient_through_all_samples = contrastive_params.gradient_through_all_samples

        self.num_negative_samples = contrastive_params.num_negative_samples
        # # (feature_dim,)
        # self.delta_feature_min = self.encoder({key: val[0] for key, val in self.params.obs_delta_range.items()})
        # self.delta_feature_max = self.encoder({key: val[1] for key, val in self.params.obs_delta_range.items()})

        self.num_pred_samples = contrastive_params.num_pred_samples
        self.num_pred_iters = contrastive_params.num_pred_iters
        self.pred_sigma_init = contrastive_params.pred_sigma_init
        self.pred_sigma_shrink = contrastive_params.pred_sigma_shrink

        self.init_model()
        self.reset_params()

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=contrastive_params.lr)

        self.load(params.training_params.load_inference, device)
        self.train()

    def init_model(self):
        raise NotImplementedError

    def reset_params(self):
        pass

    def setup_annealing(self, step):
        pass

    def cat_features(self, features, next_feature):
        """
        :param features:
            if observation space is continuous: (bs, num_observation_steps, feature_dim).
            else: [(bs, num_observation_steps, feature_i_dim)] * feature_dim
            notice that bs can be a multi-dimensional batch size
        :param next_feature:
            if observation space is continuous: (bs, feature_dim).
            else: [(bs, feature_i_dim)] * feature_dim
        :return:
            if observation space is continuous: (bs, num_observation_steps, feature_dim).
            else: [(bs, num_observation_steps, feature_i_dim)] * feature_dim
        """
        if self.continuous_state:
            features = torch.cat([features[..., 1:, :], next_feature.unsqueeze(dim=-2)], dim=-2)
        else:
            features = [torch.cat([features_i[..., 1:, :], next_feature_i.unsqueeze(dim=-2)], dim=-2)
                        for features_i, next_feature_i in zip(features, next_feature)]
        return features

    def sample_delta_feature(self, shape, num_samples):
        # (bs, num_pred_samples, feature_dim)
        uniform_noise = torch.rand(*shape, num_samples, self.feature_dim, dtype=torch.float32, device=self.device)
        delta_feature = uniform_noise * (self.delta_feature_max - self.delta_feature_min) + self.delta_feature_min
        return delta_feature

    # This is more like a temporary function that only works for collision reward
    # return # (bs, num_negative_samples, feature_dim)
    def sample_boolean_neg_feature(self, shape, reward):
        bs = shape[0]
        params = self.params
        if params.domain == "igibson":
            base = torch.ones(*shape, 1, device=self.device)
            neg = base - reward

            # # this will give collision
            # num_samples = self.num_negative_samples
            # reward_ranges = [[0, 1]]
            # assert (len(reward_ranges) == self.reward_dim)
            # neg_r_list = []
            # for r_range in reward_ranges:
            #     neg_r_list.append(np.random.choice(r_range, bs * num_samples))
            # # r_dim * (bs*mnum_samples)
            # neg_r_list = np.asarray([neg_r_list])
            # reshaped_neg_r_list = neg_r_list.T.reshape([bs, num_samples, self.reward_dim])
            # neg = torch.tensor(reshaped_neg_r_list, device=self.device)

        elif params.domain == "minigrid":
            assert(self.reward_dim == 5)

            # TODO: figure out the most efficient way
            #  OK here is a hacky implementation
            r_collision_range = [[0, 1], [-1, 1], [-1, 0]]
            idxes = reward[:,0, :2].type(torch.int) + 1

            with torch.no_grad():
                neg1 = [r_collision_range[ind] for ind in idxes[:, 0]]
                neg2 = [r_collision_range[ind] for ind in idxes[:, 1]]
                base = torch.ones(*shape, 1, device=self.device) * -5
                neg3 = base - reward[:, :, 2:]

                tf_neg1 = torch.tensor(neg1, device=self.device).unsqueeze(-1)
                tf_neg2 = torch.tensor(neg2, device=self.device).unsqueeze(-1)
                tf_neg3 = neg3.expand([-1, 2, -1])

                neg = torch.cat((tf_neg1, tf_neg2, tf_neg3), dim=2)

            # # # Previous approach to generate negative samples
            # num_samples = self.num_negative_samples
            # reward_ranges = [[-1, 0, 1], [-1, 0, 1], [0, -5], [0, -5], [0, -5]]
            # assert(len(reward_ranges) == self.reward_dim)
            # neg_r_list = []
            # for r_range in reward_ranges:
            #     neg_r_list.append(np.random.choice(r_range, bs * num_samples))
            # # r_dim * (bs*mnum_samples)
            # neg_r_list = np.asarray([neg_r_list])
            # reshaped_neg_r_list = neg_r_list.T.reshape([bs, num_samples, self.reward_dim])
            # neg = torch.tensor(reshaped_neg_r_list, device=self.device)
        return neg


    def get_feature(self, obs):
        feature = self.encoder(obs)
        return feature

    def forward_step(self, features, action, delta_features):
        """
        compute energy
        :param features:
            if observation space is continuous: (bs, num_observation_steps, feature_dim).
            else: NotImplementedError
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :param delta_features:
            if observation space is continuous: (bs, num_samples, feature_dim).
            else: NotImplementedError
        :return: energy: (bs, num_samples, feature_dim)
        """
        raise NotImplementedError

    def forward_with_feature(self, features, actions, next_features, neg_delta_features=None):
        """
        :param features:
            if observation space is continuous: (bs, num_observation_steps, feature_dim).
            else: NotImplementedError
            notice that bs can be a multi-dimensional batch size
        :param actions: (bs, num_pred_steps, action_dim)
        :param next_features:
            if observation space is continuous: (bs, num_pred_steps, feature_dim).
            else: NotImplementedError
        :param neg_delta_features:
            if observation space is continuous: (bs, num_pred_steps, num_negative_samples, feature_dim).
            else: NotImplementedError
        :return: energy:
            if observation space is continuous: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
            else: NotImplementedError
        """
        energies = []
        actions = torch.unbind(actions, dim=-2)
        next_features = torch.unbind(next_features, dim=-2)
        neg_delta_features = torch.unbind(neg_delta_features, dim=-3)
        for i, (action, next_feature, neg_delta_features) in enumerate(zip(actions, next_features, neg_delta_features)):
            delta_feature = next_feature - features[..., -1, :]                                 # (bs, feature_dim)
            delta_feature = delta_feature.unsqueeze(dim=-2)                                     # (bs, 1, feature_dim)
            # (bs, 1 + num_negative_samples, feature_dim)
            delta_features = torch.cat([delta_feature, neg_delta_features], dim=-2)
            energy = self.forward_step(features, action, delta_features)
            energies.append(energy)

            if i == len(actions) - 1:
                break

            # (bs, num_negative_samples, feature_dim)
            neg_energy = energy[..., 1:, :]
            if self.gradient_through_all_samples:
                # (bs, num_negative_samples, feature_dim)
                delta_feature_select = F.gumbel_softmax(neg_energy, dim=-2, hard=True)
                delta_feature = (neg_delta_features * delta_feature_select).sum(dim=-2)         # (bs, feature_dim)
            else:
                delta_feature_select = neg_energy.argmax(dim=-2, keepdim=True)                  # (bs, 1, feature_dim)
                delta_feature = torch.gather(neg_delta_features, -2, delta_feature_select)      # (bs, 1, feature_dim)
                delta_feature = delta_feature[..., 0, :]                                        # (bs, feature_dim)

            pred_next_feature = features[..., -1, :] + delta_feature
            features = self.cat_features(features, pred_next_feature)

        # (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        energies = torch.stack(energies, dim=-3)
        return energies

    def forward(self, obses, actions, next_obses, neg_delta_feature=None):
        features = self.get_feature(obses)
        next_features = self.get_feature(next_obses)
        return self.forward_with_feature(features, actions, next_features, neg_delta_feature)

    @staticmethod
    def nce_loss(energy):
        """
        :param energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim) or 
                       (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
        :return:
            loss: scalar
        """
        if energy.ndim == 4:
            return -F.log_softmax(energy, dim=-2)[..., 0, :].sum(dim=(-2, -1)).mean()
        elif energy.ndim == 5:
            return -F.log_softmax(energy, dim=-3)[..., 0, :, :].sum(dim=(-3, -2, -1)).mean()
        else:
            raise NotImplementedError

    def backprop(self, loss, loss_detail):
        self.optimizer.zero_grad()
        loss.backward()

        grad_clip_norm = self.contrastive_params.grad_clip_norm
        if not grad_clip_norm:
            grad_clip_norm = np.inf
        loss_detail["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)

        self.optimizer.step()
        return loss_detail

    def update(self, obses, actions, next_obses, eval=False):
        """
        :param obses: {obs_i_key: (bs, num_observation_steps, obs_i_shape)}
        :param actions: (bs, num_pred_steps, action_dim)
        :param next_obs: ({obs_i_key: (bs, num_pred_steps, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """
        bs, num_pred_steps = actions.shape[:-2], actions.shape[-2]
        # (bs, num_pred_steps, num_negative_samples, feature_dim)
        neg_delta_feature = self.sample_delta_feature(bs + (num_pred_steps,), self.num_negative_samples)
        # (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        energy = self.forward(obses, actions, next_obses, neg_delta_feature)

        loss_detail = {}

        if self.loss_type == "contrastive":
            loss = self.nce_loss(energy)
        elif self.loss_type == "mle":
            # (bs, num_pred_steps, feature_dim), (bs, num_pred_steps, num_negative_samples, feature_dim)
            pos_energy, neg_energy = energy[..., 0, :], energy[..., 1:, :]
            # (bs, num_pred_steps, num_negative_samples, feature_dim)
            neg_weight = torch.softmax(neg_energy.detach(), dim=-2)
            mle_loss = (pos_energy - (neg_weight * neg_energy).sum(dim=-2))         # (bs, num_pred_steps, feature_dim)

            energy_norm = (pos_energy ** 2 + (neg_energy ** 2).sum(dim=-2)).sum(dim=(-2, -1)).mean()
            regularization = self.l2_reg_coef * energy_norm
            
            loss = -mle_loss.sum(dim=(-2, -1)).mean() + regularization
        else:
            raise NotImplementedError

        loss_detail["contrastive_loss"] = loss

        if not eval:
            self.backprop(loss, loss_detail)

        return loss_detail

    def predict_step_with_feature(self, features, action):
        """
        :param features:
            if observation space is continuous: (bs, feature_dim).
            else: NotImplementedError
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return: pred_next_feature:
            if observation space is continuous: (bs, feature_dim).
            else: NotImplementedError
        """
        bs = action.shape[:-1]
        num_pred_samples = self.num_pred_samples
        delta_feature_max = self.delta_feature_max
        delta_feature_min = self.delta_feature_min
        sigma = self.pred_sigma_init

        delta_feature_candidates = self.sample_delta_feature(bs, num_pred_samples)
        delta_feature_candidates = torch.sort(delta_feature_candidates, dim=1)[0]

        for i in range(self.num_pred_iters):
            # (bs, num_pred_samples, feature_dim)
            if self.params.training_params.inference_algo == "contrastive_cmi":
                mask = self.get_eval_mask(bs, self.feature_dim - 1)
                forward_mode = ("causal",)

                full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy = \
                    self.forward_step(features, action, delta_feature_candidates, forward_mode)
                energy = causal_energy
            else:
                energy = self.forward_step(features, action, delta_feature_candidates)

            if i != self.num_pred_iters - 1:
                energy = energy.transpose(-2, -1)                           # (bs, feature_dim, num_pred_samples)
                dist = Categorical(logits=energy)
                idxes = dist.sample([num_pred_samples])                     # (num_pred_samples, bs, feature_dim)
                idxes = idxes.permute(*(np.arange(len(bs)) + 1), 0, -1)     # (bs, num_pred_samples, feature_dim)

                # (bs, num_pred_samples, feature_dim)
                delta_feature_candidates = torch.gather(delta_feature_candidates, -2, idxes)
                noise = torch.randn_like(delta_feature_candidates) * sigma * (delta_feature_max - delta_feature_min)
                delta_feature_candidates += noise
                delta_feature_candidates = torch.clip(delta_feature_candidates, delta_feature_min, delta_feature_max)

                sigma *= self.pred_sigma_shrink

        argmax_idx = torch.argmax(energy, dim=-2, keepdim=True)             # (bs, 1, feature_dim)
        # (bs, feature_dim)
        delta_feature = torch.gather(delta_feature_candidates, -2, argmax_idx)[..., 0, :]
        pred_next_feature = features[..., -1, :] + delta_feature

        return pred_next_feature

    def predict_with_feature(self, features, actions):
        pred_next_features = []
        for action in torch.unbind(actions, dim=-2):
            pred_next_feature = self.predict_step_with_feature(features, action)
            pred_next_features.append(pred_next_feature)
            features = self.cat_features(features, pred_next_feature)
        return torch.stack(pred_next_features, dim=-2)

    def get_adjacency(self):
        return None

    def get_intervention_mask(self):
        return None

    def train(self, training=True):
        self.training = training

    def eval(self):
        self.train(False)

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("contrastive loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

