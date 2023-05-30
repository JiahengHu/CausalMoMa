import os
import time
import json
import torch
import shutil
import random

from utils.multiprocessing_env import SubprocVecEnv
import numpy as np

import sys


class Logger(object):
    def __init__(self, fp):
        self.terminal = sys.stdout
        self.log = open(fp, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TrainingParams(AttrDict):
    def __init__(self, training_params_fname="params.json", train=True):
        config = json.load(open(training_params_fname))
        for k, v in config.items():
            self.__dict__[k] = v
        self.__dict__ = self._clean_dict(self.__dict__)

        repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        training_params = self.training_params
        if getattr(training_params, "load_inference", None) is not None:
            training_params.load_inference = \
                os.path.join(repo_path, "interesting_models", training_params.load_inference)

        if train:
            sub_dirname = "dynamics"
            info = self.info.replace(" ", "_")
            if config["train_mask"]:
                info += "_" + "train_mask"
            experiment_dirname = info + "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
            self.rslts_dir = os.path.join(repo_path, "causal", "rslts", sub_dirname, experiment_dirname)
            os.makedirs(self.rslts_dir)
            shutil.copyfile(training_params_fname, os.path.join(self.rslts_dir, "params.json"))

            self.replay_buffer_dir = None
            if training_params.replay_buffer_params.saving_freq:
                self.replay_buffer_dir = os.path.join(repo_path, "replay_buffer", experiment_dirname)
                os.makedirs(self.replay_buffer_dir)

        super(TrainingParams, self).__init__(self.__dict__)

    def _clean_dict(self, _dict):
        for k, v in _dict.items():
            if v == "":  # encode empty string as None
                v = None
            if isinstance(v, dict):
                v = self._clean_dict(v)
            _dict[k] = v
        return AttrDict(_dict)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def to_device(dictionary, device):
    """
    place dict of tensors + dict to device recursively
    """
    new_dictionary = {}
    for key, val in dictionary.items():
        if isinstance(val, dict):
            new_dictionary[key] = to_device(val, device)
        elif isinstance(val, torch.Tensor):
            new_dictionary[key] = val.to(device)
        else:
            raise ValueError("Unknown value type {} for key {}".format(type(val), key))
    return new_dictionary


def get_start_step_from_model_loading(params):
    """
    if inference is loaded, return its training step;
    else, return 0
    """
    load_inference = params.training_params.load_inference
    if load_inference is not None and os.path.exists(load_inference):
        model_name = load_inference.split(os.sep)[-1]
        start_step = int(model_name.split("_")[-1])
        print("start_step:", start_step)
    else:
        start_step = 0
    return start_step
