# DRA-GRPO
Official code for the paper: DRA-GRPO: Exploring Diversity-Aware Reward Adjustment for R1-Zero-Like Training of Large Language Models


## Installation

Clone the code. We are using the following modules.

```
module load anaconda3/2023.09-0 
module load git-lfs/3.3.0 
module load cuda/11.8.0 

```

Please follow the instructions of [Open-RS](https://github.com/knoveleng/open-rs) to install the environment.
Log in to Hugging Face and Weights & Biases:
```
huggingface-cli login
wandb login
```

```
source activate openr3
```

**You can then remove ```trl``` package from the environment, because we customized it.**



## Training

### DRA-GRPO
```
ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 11188 \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=3 \
  src/open_r1/grpo.py \
  --config recipes/dra_grpo.yaml 
```


### DRA-DR. GRPO
```
ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 18007 \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=3 \
  src/open_r1/drgrpo.py \
  --config recipes/dra_dr_grpo.yaml
```

All weights will update to Huggingface.

## Inference via lighteval (Test multiple steps)
We have an evaluation template 

```
base evaL_all.sh
```

## Checkpoints

We will release it soon!



