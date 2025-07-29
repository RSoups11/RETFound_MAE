#!/usr/bin/env python3
# optuna_search.py
# -----------------------------------------------------------
# Bayesian search on RETFound using Optuna (single GPU setup)

import os, json, shutil, warnings, datetime
import optuna
from main_finetune import get_args_parser, main

# -------- CONFIGURATION ---------------
N_TRIALS = 60
EPOCHS = 50
DB_FILE = "sqlite:///optuna/retfound.db"
DATA_PATH = "./data"
PRETRAINED_PATH = "RETFound_mae_natureCFP"
NB_CLASSES = 5
SAVE_DIR = "optuna"
# --------------------------------------

def build_args(trial):
    """Builds the argument object required by RETFound (main_finetune.py)."""
    args = get_args_parser().parse_args([])

    # ---- Optuna Search Space ----
    args.blr = trial.suggest_float("blr", 1e-5, 1e-2, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    args.drop_path = trial.suggest_float("drop_path", 0.0, 0.4)
    args.layer_decay = trial.suggest_float("layer_decay", 0.5, 0.8)
    args.smoothing = trial.suggest_float("smoothing", 0.0, 0.2)
    args.mixup = trial.suggest_float("mixup", 0.0, 0.4)
    args.cutmix = trial.suggest_float("cutmix", 0.0, 0.3)
    args.batch_size = trial.suggest_int("batch_size", 16, 32, step=16)

    # ---- Fixed Parameters ----
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
    args.savemodel = True  # required to generate checkpoint-best.pth
    args.device = "cuda"
    args.inference = False
    args.class_weighted_loss = True  # Make sure it's enabled in your code
    args.eval = False
    args.no_plots = True

    return args

def objective(trial):
    args = build_args(trial)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Starting trial #{trial.number} → Saving to {args.output_dir}")

    # Run fine-tuning and return best validation AUC
    best_auc = main(args, criterion=None)

    # Save only the best checkpoint
    if best_auc == objective.best_value:
        final_path = os.path.join(SAVE_DIR, "checkpoint-best.pth")
        shutil.copyfile(os.path.join(args.output_dir, "checkpoint-best.pth"), final_path)
        print(f"[INFO] Best checkpoint updated → {final_path}")

    return best_auc

# To track the best score across calls
objective.best_value = 0.0

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(SAVE_DIR, exist_ok=True)

    study = optuna.create_study(
        study_name="RETFound_Bayes",
        storage=DB_FILE,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True)
    )

    def wrapped_objective(trial):
        val = objective(trial)
        if val > objective.best_value:
            objective.best_value = val
        return val

    study.optimize(wrapped_objective, n_trials=N_TRIALS, n_jobs=1)

    print("\n=== SEARCH FINISHED ===")
    print("Best hyperparameters:\n", json.dumps(study.best_params, indent=2))
    print(f"Best validation AUC = {study.best_value:.4f}")
