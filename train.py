import torch
import numpy as np
import datetime
import copy
from sklearn.metrics import f1_score, precision_score, confusion_matrix
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from dataset import Dataset_ADNI_Ablation
from models import get_ablation_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import entropy
import argparse


# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────

DATA_DIR   = r'C:\Users\ravya\Desktop\AAL90'
LABEL_FILE = r'C:\Users\ravya\Desktop\label-2cls_new.csv'

ALL_MODES  = [
    'node',
    'node_edge',
    'node_edge_tri',
    'node_edge_tri_sca',
]


# ─────────────────────────────────────────────────────────────────
#  Logging helpers
# ─────────────────────────────────────────────────────────────────

def log(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")


def print_banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────

def confusion_entropy(cm):
    cm   = cm.astype(np.float32)
    cm   = cm / cm.sum()
    flat = cm.flatten()
    flat = flat[flat > 0]
    return entropy(flat, base=2)


class ModeSubset(torch.utils.data.Dataset):
    """Wraps the full dataset to expose graphs for one ablation mode."""
    def __init__(self, full_dataset, indices):
        self.full_dataset = full_dataset
        self.indices      = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.full_dataset[self.indices[i]]


# ─────────────────────────────────────────────────────────────────
#  Single ablation run (10-fold CV)
# ─────────────────────────────────────────────────────────────────

def run_ablation(args, device, dataset, mode):
    """Run 10-fold stratified CV for one ablation mode. Returns metrics dict."""
    skf = StratifiedKFold(
        n_splits=args.n_folds, random_state=args.seed, shuffle=True)

    all_acc, all_prec, all_f1, all_cm = [], [], [], []
    generator = torch.Generator().manual_seed(args.seed)
    labels    = np.array(dataset.labels)

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(labels)), labels)):

        log(f"[{mode}] Fold {fold+1}/{args.n_folds}")

        train_labels = labels[train_idx].tolist()
        val_labels   = labels[val_idx].tolist()
        log(f"  Train dist: {np.bincount(train_labels)} | "
            f"Val dist: {np.bincount(val_labels)}")

        train_data = ModeSubset(dataset, train_idx)
        val_data   = ModeSubset(dataset, val_idx)

        # Weighted loss for class imbalance
        counts        = np.bincount(train_labels)
        class_weights = torch.tensor(
            len(train_labels) / (len(counts) * counts),
            dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_loader = DataLoader(
            train_data, batch_size=args.batch_size,
            shuffle=True, generator=generator)
        val_loader = DataLoader(
            val_data, batch_size=args.batch_size,
            shuffle=False)

        model     = get_ablation_model(mode, args).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=15, min_lr=1e-6)

        best_val_loss    = float('inf')
        best_val_acc     = 0.0
        best_val_f1      = 0.0
        best_val_prec    = 0.0
        best_cm          = None
        patience_counter = 0

        for epoch in range(args.epochs):

            # ── Train ──────────────────────────────────────────
            model.train()
            tr_loss, correct, total = 0.0, 0, 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out, _  = model(data)
                loss    = criterion(out, data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()
                tr_loss += loss.item()
                preds    = out.argmax(dim=1)
                correct += (preds == data.y).sum().item()
                total   += data.y.size(0)
            tr_loss /= len(train_loader)
            tr_acc   = correct / total

            # ── Validate ───────────────────────────────────────
            model.eval()
            vl_loss, vl_correct, vl_total = 0.0, 0, 0
            vl_preds, vl_trues = [], []
            with torch.no_grad():
                for data in val_loader:
                    data    = data.to(device)
                    out, _  = model(data)
                    loss    = criterion(out, data.y)
                    vl_loss += loss.item()
                    preds   = out.argmax(dim=1)
                    vl_correct += (preds == data.y).sum().item()
                    vl_total   += data.y.size(0)
                    vl_preds.extend(preds.cpu().numpy())
                    vl_trues.extend(data.y.cpu().numpy())

            vl_loss /= len(val_loader)
            vl_acc   = vl_correct / vl_total
            vl_f1    = f1_score(
                vl_trues, vl_preds, average='weighted', zero_division=0)
            vl_prec  = precision_score(
                vl_trues, vl_preds, average='weighted', zero_division=0)
            cm       = confusion_matrix(vl_trues, vl_preds)

            scheduler.step(vl_loss)

            log(f"  Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} | "
                f"Val Loss:{vl_loss:.4f} Acc:{vl_acc:.4f} | "
                f"F1:{vl_f1:.4f}")

            # Early stopping
            if vl_loss < best_val_loss:
                best_val_loss    = vl_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    log(f"  Early stopping at epoch {epoch+1}")
                    break

            # Track best accuracy checkpoint
            if vl_acc > best_val_acc:
                best_val_acc  = vl_acc
                best_val_f1   = vl_f1
                best_val_prec = vl_prec
                best_cm       = cm

        cm_str = np.array2string(
            best_cm, separator=' ',
            max_line_width=np.inf).replace('\n', ' ')
        log(f"  Fold {fold+1} Best → "
            f"Acc={best_val_acc:.4f} "
            f"Prec={best_val_prec:.4f} "
            f"F1={best_val_f1:.4f}")
        log(f"  Fold {fold+1} CM: {cm_str}")

        all_acc.append(best_val_acc)
        all_prec.append(best_val_prec)
        all_f1.append(best_val_f1)
        all_cm.append(best_cm)

        log(f"  Running avg Fold {fold+1}: "
            f"Acc={np.mean(all_acc):.4f} "
            f"Prec={np.mean(all_prec):.4f} "
            f"F1={np.mean(all_f1):.4f}")

    final_acc  = float(np.mean(all_acc))
    final_prec = float(np.mean(all_prec))
    final_f1   = float(np.mean(all_f1))
    avg_cm     = np.mean(all_cm, axis=0)
    ent        = confusion_entropy(avg_cm)

    log(f"FINAL [{mode}] → "
        f"Acc={final_acc:.4f} Prec={final_prec:.4f} "
        f"F1={final_f1:.4f} Entropy={ent:.4f}")

    return {
        'mode':          mode,
        'accuracy':      final_acc,
        'precision':     final_prec,
        'f1':            final_f1,
        'entropy':       ent,
        'conf_matrix':   avg_cm,
        'per_fold_acc':  all_acc,
        'per_fold_f1':   all_f1,
        'per_fold_prec': all_prec,
    }


# ─────────────────────────────────────────────────────────────────
#  Main training entry point
# ─────────────────────────────────────────────────────────────────

def train(args, device):
    # Load dataset once — graphs precomputed per subject
    dataset = Dataset_ADNI_Ablation(
        data_dir    = DATA_DIR,
        label_file  = LABEL_FILE,
        num_classes = args.num_classes,
        k_neighbors = 12,
        K_bins      = 5,
    )

    results = {}
    for mode in ALL_MODES:
        print_banner(f"ABLATION: {mode.upper()}")
        results[mode] = run_ablation(args, device, dataset, mode)

    # ── Final summary table ────────────────────────────────────
    print_banner("FINAL ABLATION SUMMARY")

    col_w = 22
    print(f"{'Mode':<{col_w}} {'Accuracy':>10} {'Precision':>10} "
          f"{'F1':>10} {'Entropy':>10}")
    print("-" * (col_w + 42))
    for mode in ALL_MODES:
        r = results[mode]
        print(f"{r['mode']:<{col_w}} "
              f"{r['accuracy']:>10.4f} "
              f"{r['precision']:>10.4f} "
              f"{r['f1']:>10.4f} "
              f"{r['entropy']:>10.4f}")

    print(f"\nPer-fold Accuracy across ablation modes:")
    header = f"  {'Fold':<6}" + "".join(
        f"{m:>{col_w}}" for m in ALL_MODES)
    print(header)
    print("  " + "-" * (6 + col_w * len(ALL_MODES)))
    for i in range(args.n_folds):
        row = f"  {i+1:<6}"
        for m in ALL_MODES:
            row += f"{results[m]['per_fold_acc'][i]:>{col_w}.4f}"
        print(row)
    mean_row = f"  {'Mean':<6}"
    for m in ALL_MODES:
        mean_row += f"{results[m]['accuracy']:>{col_w}.4f}"
    print(mean_row)
    print()


# ─────────────────────────────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='HOCFormer Ablation Training')
    parser.add_argument('--num_classes',   type=int,   default=2)
    parser.add_argument('--hidden_dim',    type=int,   default=256)
    parser.add_argument('--n_layers',      type=int,   default=3)
    parser.add_argument('--dropout',       type=float, default=0.5)
    parser.add_argument('--n_folds',       type=int,   default=10)
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--seed',          type=int,   default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    log(f"Args: {vars(args)}")
    train(args, device)
