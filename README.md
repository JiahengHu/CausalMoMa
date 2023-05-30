# Causal Policy Gradient for Whole-Body Mobile Manipulation

Jiaheng Hu, Peter Stone, Roberto Martin-Martin

RSS2023

## Setup

1. Clone this repo and its submodules:
```bash
git clone https://github.com/JiahengHu/CausalMoMa.git --recursive
```

2. Install the cloned `iGibson-CausalMoMa`, `Minigrid-CausalMoMa` and `sb3-CausalMoMa`, following the respective `README.md` instructions.

3. Download the required [iGibson data](https://stanfordvl.github.io/iGibson/installation.html#downloading-the-assets-and-datasets-of-scenes-and-objects). 
Download [HSR mesh data](https://drive.google.com/file/d/1Vz-Shra-Y3ZiHJdCjnQg8hZBqFPw6byG/view?usp=sharing) and extract 
it into `iGibson-CausalMoMa/igibson/data/assets/models/hsr`

## Causal Inference
1. Download pre-collected [Causal inference data](https://drive.google.com/drive/folders/1j0sSoHC_Hx6Wcel4mDvBevXboYOg1dKs?usp=sharing) and put them into `data/`.
Alternatively, collect new data by running:
```
# iGibson data
python collect_igibson_data.py

# Minigrid data
python collect_igibson_data.py
```


2. Run causal discovery with one of the config file provided:
```
python causal_inference.py --config PATH_TO_CONFIG

# e.g., for minigrid
python causal_inference.py --config configs/minigrid_full.json
```

Results will be stored inside `causal/`.

## Policy Learning

The inferred causal matrix is already put inside `train.py`

```
# HSR with Causal MoMa
python train.py -sc --robot hsr

# HSR with Vanilla PPO
python train.py -fc --robot hsr

# Fetch with Causal MoMa
python train.py -sc --robot fetch

# Fetch with Vanilla PPO
python train.py -fc --robot fetch
```

Results will be stored inside `log_dir/`.

## Citing
```
@inproceedings{hu2023causal,
  title={Causal Policy Gradient for Whole-Body Mobile Manipulation},
  author={Hu, Jiaheng and Stone, Peter and Mart{\'\i}n-Mart{\'\i}n, Roberto},
  booktitle={arXiv preprint arXiv:2305.04866},
  year={2023}
}
```
