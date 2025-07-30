#!/usr/bin/env python3
# optuna_search.py
# -----------------------------------------------------------
# Bayesian search on RETFound using Optuna (single GPU setup)
# Includes pruning + logging + disk management

import os
import json
import shutil
import csv
import warnings
import optuna
from datetime import datetime
from main_finetune import get_args_parser, main

# -------- CONFIGURATION ---------------
N_TRIALS = 40
EPOCHS = 100
DB_FILE = "sqlite:///optuna/retfound.db"
DATA_PATH = "./data"
PRETRAINED_PATH = "RETFound_mae_natureCFP"  # or path to your RETFound_shanghai.pth
NB_CLASSES = 2
SAVE_DIR = "optuna"
RESULTS_CSV = os.path.join(SAVE_DIR, "optuna_results.csv")
# --------------------------------------

# Global tracker
objective_best_score = {"score": 0.0, "trial_id": -1}

csv_fields = [
    "trial_id", "auc",
    "blr", "weight_decay", "drop_path", "layer_decay",
    "smoothing", "mixup", "cutmix", "batch_size"
]

def build_args(trial):
    """Builds argparse arguments for RETFound fine-tuning"""
    args = get_args_parser().parse_args([])

    # --- Search space ---
    args.blr = trial.suggest_float("blr", 0.00214, 0.00643, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 0.00044, 0.00133, log=True)
    args.drop_path = trial.suggest_float("drop_path", 0.014 , 0.054)
    args.layer_decay = trial.suggest_float("layer_decay", 0.592, 0.693)
    args.smoothing = trial.suggest_float("smoothing", 0.080, 0.180)
    args.mixup = trial.suggest_float("mixup", 0.056, 0.156)
    args.cutmix = trial.suggest_float("cutmix", 0.124, 0.224)
    args.batch_size = 32

    # --- Fixed args ---
    args.model = "RETFound_mae"
    args.epochs = EPOCHS
    args.input_size = 224
    args.nb_classes = NB_CLASSES
    args.data_path = DATA_PATH
    args.finetune = PRETRAINED_PATH
    args.global_pool = True
    args.reprob = 0.25
    args.task = f"optuna_trial_{trial.number}"
    args.output_dir = os.path.join(SAVE_DIR, f"trial_{trial.number}")
    args.log_dir = "./output_logs"
    args.savemodel = True
    args.device = "cuda"
    args.inference = False
    # args.class_weighted_loss = True # Make sure it's enabled in your code
    args.eval = False
    args.no_plots = True

    return args

def log_trial_to_csv(trial, auc):
    """Logs hyperparameters and AUC to CSV"""
    row = {
        "trial_id": trial.number,
        "auc": auc,
        "blr": trial.params["blr"],
        "weight_decay": trial.params["weight_decay"],
        "drop_path": trial.params["drop_path"],
        "layer_decay": trial.params["layer_decay"],
        "smoothing": trial.params["smoothing"],
        "mixup": trial.params["mixup"],
        "cutmix": trial.params["cutmix"],
        "batch_size": trial.params["batch_size"]
    }

    new_file = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

def objective(trial):
    args = build_args(trial)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Starting trial #{trial.number} → Saving to {args.output_dir}")
    auc = main(args, criterion=None)

    # Report intermediate score for pruning
    trial.report(auc, step=args.epochs)
    if trial.should_prune():
        print(f"[PRUNE] Trial #{trial.number} pruned at epoch {args.epochs} with AUC={auc:.4f}")
        raise optuna.exceptions.TrialPruned()

    # Log result to CSV
    log_trial_to_csv(trial, auc)

    # Checkpoint paths
    ckpt_src = os.path.join(args.output_dir, "checkpoint-best.pth")
    ckpt_dest = os.path.join(SAVE_DIR, "checkpoint-best.pth")

    # Save checkpoint only if it's the best so far
    if auc > objective_best_score["score"]:
        if os.path.exists(ckpt_src):
            shutil.copyfile(ckpt_src, ckpt_dest)
            print(f"[INFO] New best model saved at: {ckpt_dest}")
        objective_best_score["score"] = auc
        objective_best_score["trial_id"] = trial.number
    else:
        # If not the best, delete the checkpoint to save space
        if os.path.exists(ckpt_src):
            os.remove(ckpt_src)

    # Remove the trial directory regardless
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    return auc

# ------------- ENTRYPOINT -------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(SAVE_DIR, exist_ok=True)

    study = optuna.create_study(
        study_name="RETFound_Bayes",
        storage=DB_FILE,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    )

    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    print("\n=== OPTUNA SEARCH COMPLETE ===")
    print("Best trial ID       :", objective_best_score["trial_id"])
    print("Best validation AUC :", objective_best_score["score"])
    print("Best hyperparameters:\n", json.dumps(study.best_params, indent=2))
