# VizDoomDSAC_V2# ViZDoom + DSAC-v2 Minimal Setup Guide

## Reference

* [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
* [Distributional Soft Actor-Critic v2 (DSAC-v2)](https://github.com/Jingliang-Duan/DSAC-v2)

## Requires

* Windows 7 or greater or Linux
* Python 3.8+
* Please ensure the installation path does **not** include non-English characters (especially for Windows)

## Installation

### Step 1: Clone ViZDoom

```bash
git clone https://github.com/Farama-Foundation/ViZDoom.git
cd ViZDoom
```

### Step 2: Clone DSAC-v2 into ViZDoom

```bash
git clone https://github.com/Jingliang-Duan/DSAC-v2.git
```

### Step 3: Create Conda Environment (recommended)

Make sure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

```bash
cd DSAC-v2
conda env create -f DSAC_gpu_environment.yml
conda activate DSAC_gpu
```

## Train

Run the following command inside `DSAC-v2` directory to start training:

```bash
python DSAC_V2.py
```

After training, logs and reward graphs will be generated:

* `./log/` for logs
* `./reward_plot.png` for reward plot

## Customization

### Change Scenario

Modify the following in `DSAC_V2.py`:

```python
config_path = os.path.join(current_dir, "scenarios", "defend_the_center.cfg")
wad_path = os.path.join(current_dir, "scenarios", "defend_the_center.wad")
```

### Adjust Reward Function

Edit `get_reward()` or reward-related logic in the environment wrapper.

### Modify Hyperparameters

```python
LEARNING_RATE = 1e-5  
GAMMA = 0.99
NUM_EPISODES = 2000000
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50000
UPDATE_FREQ = 4
EPSILON_START = 1.0
EPSILON_END = 0.2
EPSILON_DECAY = 0.9995
MAX_HEALTH = 100
FORWARD_REWARD = 0.1  
BACKWARD_PENALTY = 0.02  
HEALTH_REWARD = 1.0  
DAMAGE_PENALTY = 1.0  
KILL_REWARD = 500.0  
SURVIVAL_REWARD = 0.2  
ROTATION_PENALTY = 0.00001  
ATTACK_REWARD = 5.0  
DODGE_REWARD = 2.0  
HEALTH_THRESHOLD = 30
DEATH_PENALTY = 100.0 
```

## Notes

* For Windows users, if any DLL error occurs, consider using WSL or Conda.
* Ensure correct CUDA driver/toolkit installed if using GPU.

## Acknowledgment

Thanks to authors of DSAC-v2 and the ViZDoom community.
