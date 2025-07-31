#!/usr/bin/env python3
# optuna_search.py - quiet version

import os, sys, json, shutil, csv, warnings, contextlib, logging
from pathlib import Path
from datetime import datetime
import optuna

from main_finetune import get_args_parser, main as finetune_main

# ----------------------------- CONFIG ---------------------------------- #
N_TRIALS      = 40
EPOCHS        = 100
SAVE_ROOT     = Path("optuna")
DB_FILE       = f"sqlite:///{SAVE_ROOT}/retfound.db"
# DB_FILE = "sqlite:///:memory:" # keep it in RAM instead
DATA_PATH     = "./data"
PRETRAINED    = "RETFound_mae_natureCFP"
NB_CLASSES    = 2
RESULTS_CSV   = SAVE_ROOT / "optuna_results.csv"

SAVE_ROOT.mkdir(exist_ok=True, parents=True)
# ----------------------------------------------------------------------- #

# ---------------------- Logger Optuna ----------------------- #
log_path = SAVE_ROOT / "optuna.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8")]
)
optuna.logging.set_verbosity(optuna.logging.INFO)
optuna.logging.enable_propagation()   # route tout vers logging.root
# ----------------------------------------------------------------------- #

# Global best
best_model = {"score": 0.0, "path": None, "trial": -1}

csv_fields = [
    "trial_id", "auc",
    "blr", "weight_decay", "drop_path", "layer_decay",
    "smoothing", "mixup", "cutmix", "batch_size"
]

def build_args(trial):
    args = get_args_parser().parse_args([])

    # search-space
    args.blr          = trial.suggest_float("blr", 0.00214, 0.00643, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 0.00044, 0.00133, log=True)
    args.drop_path    = trial.suggest_float("drop_path", 0.014, 0.054)
    args.layer_decay  = trial.suggest_float("layer_decay", 0.592, 0.693)
    args.smoothing    = trial.suggest_float("smoothing", 0.080, 0.180)
    args.mixup        = trial.suggest_float("mixup", 0.056, 0.156)
    args.cutmix       = trial.suggest_float("cutmix", 0.124, 0.224)
    args.batch_size   = 32
    # fixed
    args.model        = "RETFound_mae"
    args.epochs       = EPOCHS
    args.input_size   = 224
    args.nb_classes   = NB_CLASSES
    args.data_path    = DATA_PATH
    args.finetune     = PRETRAINED
    args.global_pool  = True
    args.reprob       = 0.25
    args.output_dir   = str(SAVE_ROOT / f"trial_{trial.number}")
    args.log_dir      = None           # deactivate TensorBoard
    args.savemodel    = True
    args.device       = "cuda"
    # args.class_weighted_loss = True
    args.inference = False
    args.no_plots     = True
    return args

def manage_best_checkpoint(auc, trial_dir: Path):
    """If auc > global value, copy the checkpoint in optuna/best_checkpoint.pth."""
    ckpt_src = trial_dir / "checkpoint-best.pth"
    if not ckpt_src.exists():
        return

    global best_model
    if auc > best_model["score"]:
        dest = SAVE_ROOT / "best_checkpoint.pth"
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(ckpt_src, dest)
        # delete the old .pth
        if best_model["path"] and best_model["path"].exists():
            best_model["path"].unlink(missing_ok=True)
        best_model.update(score=auc, path=dest, trial=trial_dir.name)


def log_csv(trial_id, auc, params):
    new_file = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "trial_id": trial_id, "auc": auc, "batch_size": 32, **params
        })

def objective(trial):
    args = build_args(trial)
    trial_dir = Path(args.output_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    try:
        # MUTE EXEC
        with open(os.devnull, "w") as devnull, \
             contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
            auc = finetune_main(args, criterion=None)

        # Pruning
        trial.report(auc, step=args.epochs)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # CSV + checkpoint if non-pruned
        log_csv(trial.number, auc, trial.params)
        manage_best_checkpoint(auc, trial_dir)

        return auc

    finally:
        # Force clean of trials folders
        shutil.rmtree(trial_dir, ignore_errors=True)

# ----------------------------- RUN ------------------------------------- #
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    study = optuna.create_study(
        study_name="RETFound_Bayes",
        direction="maximize",
        storage=DB_FILE,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    logging.info("Search complete – best AUC %.4f (trial %d)",
                 best_model["score"], best_model["trial"])
    logging.info("Best params: %s", json.dumps(study.best_params))
