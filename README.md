# Efficient RL via Disentangled Environment and Agent Representations

This is an implementation of SEAR (Structured Environment-Agent Representations) from [Efficient RL via Disentangled Environment and Agent Representations](https://sear-rl.github.io/), by Kevin Gmelin, Shikhar Bahl, Russell Mendonca, and Deepak Pathak.

This repo was initially forked from the original [DrQ-v2 repo](https://github.com/facebookresearch/drqv2). 

## Instructions

Install mujoco:
```sh
mkdir ~/.mujoco
wget -P ~/.mujoco https://www.roboti.us/file/mjkey.txt
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
mv mujoco200_linux ~/.mujoco/mujoco200
rm mujoco200_linux.zip
```

Export the following variables (It is recommended to put this into your bashrc or zshrc file)
```sh
export MUJOCO_PY_MJKEY_PATH=~/.mujoco/mjkey.txt
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco200
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin:/usr/lib/nvidia-000
```

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Install dependencies:
```sh
# Clone this repo
git clone git@github.com:sear-rl/sear-rl.git

# We use a fork of metaworld that adds in the ability to render segmented images of the environment
git clone git@github.com:KevinGmelin/metaworld.git

# Create a conda environment with all of the dependencies except for metaworld
cd sear-rl
conda env create -f conda_env.yml
conda activate sear

# Install metaworld
cd ../metaworld
pip install -e .

# Install this repo so that imports will work properly
cd ../sear-rl
pip install -e .

# Download the DAVIS dataset if you plan on using the distracting-control suite.
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip
rm DAVIS-2017-trainval-480p.zip
```

Train the agent:
```sh
python sear/train.py task=metaworld_hammer-v2
```

To use WandB:
```sh
python sear/train.py task=metaworld_hammer-v2 use_wandb=true wandb.run_name='WandB-Run-Name’
```

To use a different agent:
```sh
python sear/train.py task=metaworld_pick-place-v2 agent=sear
```

Monitor results:
```sh
tensorboard --logdir exp_local
```

## Citation
If you use this repo, please cite our paper

```
@InProceedings{pmlr-v202-gmelin23a,
  title = {Efficient {RL} via Disentangled Environment and Agent Representations},
  author = {Gmelin, Kevin and Bahl, Shikhar and Mendonca, Russell and Pathak, Deepak},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {11525--11545},
  year = {2023},
  editor = {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, 
            Sivan and Scarlett, Jonathan},
  volume = {202},
  series = {Proceedings of Machine Learning Research},
  month = {23--29 Jul},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v202/gmelin23a/gmelin23a.pdf},
  url = {https://proceedings.mlr.press/v202/gmelin23a.html},
}
```

Also, please cite the original DrQv2 paper, upon which the corresponding code repo was started from:

```
@article{yarats2021drqv2,
  title={Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning},
  author={Denis Yarats and Rob Fergus and Alessandro Lazaric and Lerrel Pinto},
  journal={arXiv preprint arXiv:2107.09645},
  year={2021}
}
```

## License
The majority of DrQ-v2, which the SEAR codebase was initially started from, as well as all changes introduced in developing SEAR, are licensed under the MIT license, however portions of the project are available under separate license terms: DeepMind is licensed under the Apache 2.0 license.
