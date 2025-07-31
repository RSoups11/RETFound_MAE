# RETFound Fine-Tuning Usage Guide

This guide explains how to use the `main_finetune.py` script to fine-tune the RETFound model on a classification task using our own dataset.

## Example command train (multi-class with class_weighted_loss)

```bash
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --model RETFound_mae \
    --savemodel \
    --global_pool \
    --batch_size 32 \
    --epochs 50 \
    --blr 6e-4 --layer_decay 0.8 \
    --weight_decay 0.1 --drop_path 0.1 \
    --nb_classes 5 \
    --input_size 224 \
    --data_path ./data \
    --task <task_name> \
    --finetune RETFound_mae_natureCFP \
    --class_weighted_loss
```
## Example command train (binary classification)

```bash
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --model RETFound_mae \
    --savemodel \
    --global_pool \
    --batch_size 32 \
    --epochs 50 \
    --blr 6e-4 --layer_decay 0.8 \
    --weight_decay 0.1 --drop_path 0.1 \
    --nb_classes 2 \
    --input_size 224 \
    --data_path ./data \
    --task <task_name> \
    --finetune RETFound_mae_natureCFP \
```

## Example command eval

```bash
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
      --model RETFound_mae \
      --eval \
      --global_pool \
      --batch_size 32 \
      --nb_classes 5 \
      --input_size 224 \
      --data_path ./data \
      --task <task_name> \
      --best_threshold 0.2 \
      --resume ./output_dir/<task_name>/checkpoint-best.pth
```

## Example command inference

```bash
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
  --model RETFound_mae \
  --eval \
  --global_pool \
  --batch_size 32 \
  --nb_classes 5 \
  --input_size 224 \
  --data_path ./data \
  --task <task_name> \
  --resume ./output_dir/<task_name>/checkpoint-best.pth
```

## Arguments

### General Training Parameters

* `--batch_size`: Batch size per GPU (default: `128`).
* `--epochs`: Number of epochs to train (default: `50`).
* `--accum_iter`: Gradient accumulation steps (default: `1`).

### Model Parameters

* `--model`: Model backbone to use for fine-tuning (e.g., `RETFound_mae` or `Dino_V2`).
* `--input_size`: Image input size in pixels (default: `256`).
* `--drop_path`: Drop path rate (default: `0.2`).
* `--best_threshold`: Threshold for binary classification only (default: `None`).

### Optimizer Parameters

* `--clip_grad`: Maximum norm for gradient clipping (default: `None`).
* `--weight_decay`: Weight decay for regularization (default: `0.05`).
* `--lr`: Absolute learning rate (overrides `blr` if provided).
* `--blr`: Base learning rate (default: `5e-3`). Final LR is scaled by batch size.
* `--layer_decay`: Layer-wise learning rate decay factor (default: `0.65`).
* `--min_lr`: Minimum LR allowed by the scheduler (default: `1e-6`).
* `--warmup_epochs`: Number of epochs for learning rate warmup (default: `10`).

### Data Augmentation

* `--color_jitter`: Color jitter factor (default: `None`).
* `--aa`: AutoAugment policy string (default: `'rand-m9-mstd0.5-inc1'`).
* `--smoothing`: Label smoothing factor (default: `0.1`).

#### Random Erase Parameters

* `--reprob`: Probability of applying random erase (default: `0.25`).
* `--remode`: Mode for erase (default: `'pixel'`).
* `--recount`: Number of erase ops per image (default: `1`).
* `--resplit`: Use clean augmentation split before erase (default: `False`).

#### Mixup Parameters

* `--mixup`: Mixup alpha value (default: `0`).
* `--cutmix`: CutMix alpha value (default: `0`).
* `--cutmix_minmax`: Min/max ratio for CutMix (default: `None`).
* `--mixup_prob`: Probability to apply mixup or cutmix (default: `1.0`).
* `--mixup_switch_prob`: Probability to switch between mixup and cutmix (default: `0.5`).
* `--mixup_mode`: How to apply mixup: `batch`, `pair`, or `elem` (default: `batch`).

### Fine-Tuning Parameters

* `--finetune`: Path to pretrained model checkpoint (e.g., `RETFound_mae_natureCFP`).
* `--task`: Task name used for logging and saving outputs.
* `--global_pool`: Use global average pooling instead of class token.
* `--cls_token`: Use class token (overrides `--global_pool`).

### Dataset Parameters

* `--data_path`: Path to dataset root folder (default: `./data/`).
* `--nb_classes`: Number of output classes (e.g., `5`).
* `--output_dir`: Directory to save model checkpoints (default: `./output_dir`).
* `--log_dir`: Directory for TensorBoard logs (default: `./output_logs`).
* `--device`: Device to use for training (default: `cuda`).
* `--seed`: Random seed (default: `0`).
* `--resume`: Resume training from a checkpoint.
* `--start_epoch`: Epoch to start from (default: `0`).
* `--eval`: Run evaluation only.
* `--dist_eval`: Enable distributed evaluation (default: `False`).
* `--num_workers`: Number of data loading workers (default: `10`).
* `--pin_mem`: Pin CPU memory in dataloader (enabled by default).

### Distributed Training

* `--world_size`: Number of distributed processes (default: `1`).
* `--local_rank`: Local rank (for multi-GPU training).
* `--dist_on_itp`: Internal flag (leave default).
* `--dist_url`: URL used to set up distributed training (default: `env://`).

### Custom RETFound Parameters

* `--savemodel`: Save the best model (default: `True`).
* `--norm`: Image normalization mode (default: `'IMAGENET'`).
* `--enhance`: Use enhanced data if applicable (default: `False`).
* `--datasets_seed`: Random seed for splitting datasets (default: `2026`).
* `--no_plots`: Skip generation of training/ROC plots (default: `False`).
* `--class_weighted_loss`: Use class weights in loss function (Inversed frequency formula).
* `--inference`: Activate inference mode to provide a csv of the prediction (use when labell).
