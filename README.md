# DRA-GRPO 
Official code for the paper: DRA-GRPO: Exploring Diversity-Aware Reward Adjustment for R1-Zero-Like Training of Large Language Models [![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2505.09655)

Paper link (preprint): https://arxiv.org/abs/2505.09655



> **Abstract.** Recent advances in reinforcement learning for language model post-training, such as Group Relative Policy Optimization (GRPO), have shown promise in low-resource settings. However, GRPO typically relies on solution-level and scalar reward signals that fail to capture the semantic diversity among sampled completions. This leads to what we identify as a diversity-quality inconsistency, where distinct reasoning paths may receive indistinguishable rewards. To address this limitation, we propose $\textit{Diversity-aware Reward Adjustment} (DRA)$, a method that explicitly incorporates semantic diversity into the reward computation. DRA uses Submodular Mutual Information (SMI) to downweight redundant completions and amplify rewards for diverse ones. This encourages better exploration during learning, while maintaining stable exploitation of high-quality samples. Our method integrates seamlessly with both GRPO and its variant DR.~GRPO, resulting in $\textit{DRA-GRPO}$ and $\textit{DGA-DR.~GRPO}$. We evaluate our method on five mathematical reasoning benchmarks and find that it outperforms recent strong baselines. It achieves state-of-the-art performance with an average accuracy of 58.2\%, using only 7,000 fine-tuning samples and a total training cost of approximately $55.

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



