import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score, roc_curve
)
from pycm import ConfusionMatrix
import util.misc as misc
import util.lr_sched as lr_sched
import pandas as pd

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()
    
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    # for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, _, targets = batch
        # samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        loss /= accum_iter
        
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """Evaluate the model."""
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    model.eval()
    score = float("nan")
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []
    img_paths, y_true_list, y_pred_list, y_prob1_list = [], [], [], []
    
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        # images, target = batch[0].to(device, non_blocking=True), batch[-1].to(device, non_blocking=True)
        images, paths, target = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        output_ = nn.Softmax(dim=1)(output)
        proba1  = output_[:, 1]
        if num_class == 2 and getattr(args, "best_threshold", None) is not None:
            output_label = (proba1 >= args.best_threshold).long()
        else:
            output_label = output_.argmax(dim=1)
        output_onehot = F.one_hot(output_label.to(torch.int64), num_classes=num_class)
        
        metric_logger.update(loss=loss.item())
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot.extend(output_onehot.detach().cpu().numpy())
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(output_label.detach().cpu().numpy())
        pred_softmax.extend(output_.detach().cpu().numpy())

        img_paths.extend(paths)
        y_true_list.extend(target.cpu().numpy())
        y_pred_list.extend(output_label.cpu().numpy())
        y_prob1_list.extend(proba1.cpu().numpy())

    if not args.no_plots:
        if num_class == 2 and mode in ['val', 'test']:
            y_true = np.array(true_labels)
            y_probs = np.array(pred_softmax)[:, 1]  # proba classe 1

            if len(np.unique(y_true)) < 2:
                print("[WARNING] ROC AUC not computed: only one class present in y_true")
                auc_score = float("nan")
            else:
                fpr, tpr, thresholds = roc_curve(y_true, y_probs)
                auc_score = roc_auc_score(y_true, y_probs)
                
                plt.figure(figsize=(8, 5), dpi=150)
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve - {mode}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
            
                fig_path = os.path.join(args.output_dir, args.task, f'roc_curve_{mode}.png')
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close()
                
    safe_metrics = True
    if mode == 'test' and len(np.unique(true_labels)) < 2:
        print("[WARNING] Skipping full metric computation: only one class in ground-truth.")
        safe_metrics = False

    if safe_metrics:
        accuracy = accuracy_score(true_labels, pred_labels)
        hamming = hamming_loss(true_onehot, pred_onehot)
        jaccard = jaccard_score(true_onehot, pred_onehot, average='macro')
        average_precision = average_precision_score(true_onehot, pred_softmax, average='macro')
        kappa = cohen_kappa_score(true_labels, pred_labels)
        f1 = f1_score(true_onehot, pred_onehot, zero_division=0, average='macro')
        roc_auc = roc_auc_score(true_onehot, pred_softmax, multi_class='ovr', average='macro')
        precision = precision_score(true_onehot, pred_onehot, zero_division=0, average='macro')
        recall = recall_score(true_onehot, pred_onehot, zero_division=0, average='macro')
        
        score = (f1 + roc_auc + kappa) / 3
        if log_writer:
            for metric_name, value in zip(['accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa', 'score'],
                                           [accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa, score]):
                log_writer.add_scalar(f'perf/{metric_name}', value, epoch)
        
        print(f'val loss: {metric_logger.meters["loss"].global_avg}')
        print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, Hamming Loss: {hamming:.4f},\n'
              f' Jaccard Score: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},\n'
              f' Average Precision: {average_precision:.4f}, Kappa: {kappa:.4f}, Score: {score:.4f}')
        
        metric_logger.synchronize_between_processes()
        
        results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')
        file_exists = os.path.isfile(results_path)
        with open(results_path, 'a', newline='', encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            if not file_exists:
                wf.writerow(['val_loss', 'accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa'])
            wf.writerow([metric_logger.meters["loss"].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa])

    else :
        score = float("nan")

    if mode == 'test':
        print("[INFO] Test mode: generating confusion matrix.")
    
        # Plot Confusion Matrix
        try:
            cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)
            cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
            plt.savefig(os.path.join(args.output_dir, args.task, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')
            print("[INFO] Confusion matrix saved.")
        except Exception as e:
            print(f"[WARNING] Failed to generate confusion matrix: {e}")

    # Save predictions when inference
    if getattr(args, 'inference', False):
        print("[INFO] Inference mode: saving detailed predictions.")

        if num_class == 2:
            df_preds = pd.DataFrame({
                "filepath": img_paths,
                "y_true": y_true_list,
                "y_pred": y_pred_list,
                "proba_0": [round(float(1 - prob), 4) for prob in y_prob1_list],
                "proba_1": [round(float(prob), 4) for prob in y_prob1_list],
            })
        else:
            # Multi-class
            proba_cols = {
                f"proba_{i}": [round(float(probs[i]), 4) for probs in pred_softmax]
                for i in range(num_class)
            }
            df_preds = pd.DataFrame({
                "filepath": img_paths,
                "y_true": y_true_list,
                "y_pred": y_pred_list,
                **proba_cols
            })

        df_preds.to_csv(
            os.path.join(args.output_dir, args.task, f"predictions.csv"),
            index=False
        )
        print("[INFO] Predictions saved.")
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, score
