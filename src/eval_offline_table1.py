import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoTokenizer
except Exception as e:
    raise RuntimeError("transformers not installed. Please install transformers first.") from e

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.ulapm import build_ulapm_from_artifacts


def acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred, num_classes=None):
    try:
        from sklearn.metrics import f1_score

        return float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if num_classes is None:
            num_classes = int(max(y_true.max(), y_pred.max()) + 1)
        f1s = []
        for c in range(num_classes):
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1s.append(2 * prec * rec / (prec + rec + 1e-12))
        return float(np.mean(f1s))


def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(((a - b) ** 2).mean())


def ccc(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    return float((2 * cov) / (vx + vy + (mx - my) ** 2 + 1e-12))


def _normalize_probs(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    p /= p.sum(axis=-1, keepdims=True)
    return p


def behavior_kl(p_true, p_pred, eps=1e-12):
    p = _normalize_probs(p_true, eps=eps)
    q = _normalize_probs(p_pred, eps=eps)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))


def behavior_js(p_true, p_pred, eps=1e-12):
    p = _normalize_probs(p_true, eps=eps)
    q = _normalize_probs(p_pred, eps=eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=-1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=-1)
    return float(np.mean(0.5 * kl_pm + 0.5 * kl_qm))


def load_split(split_path):
    sp = np.load(split_path, allow_pickle=True)
    files = set(sp.files)
    if {"train_idx", "val_idx", "test_idx"}.issubset(files):
        return sp["train_idx"], sp["val_idx"], sp["test_idx"]
    if {"train", "val", "test"}.issubset(files):
        return sp["train"], sp["val"], sp["test"]
    raise KeyError(f"Split file keys not recognized. got={sp.files}")


class NPZTextDataset(Dataset):
    def __init__(self, npz_path, indices, tokenizer_name="distilbert-base-uncased", max_len=128, local_files_only=False):
        data = np.load(npz_path, allow_pickle=True)
        self.texts = data["texts"]
        self.c = data["c"].astype(np.int64)
        self.va = data["va"].astype(np.float32)
        self.b = data["b"].astype(np.float32)
        self.idx = indices.astype(np.int64)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)
        self.max_len = int(max_len)

    def __len__(self):
        return int(len(self.idx))

    def __getitem__(self, i):
        j = int(self.idx[i])
        text = str(self.texts[j])
        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "c": torch.tensor(self.c[j], dtype=torch.long),
            "va": torch.tensor(self.va[j], dtype=torch.float32),
            "b": torch.tensor(self.b[j], dtype=torch.float32),
        }


def load_state_dict_robust(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        return sd["model"]
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="distilbert-base-uncased")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = torch.device("cpu")
    print("device:", device)

    _, _, test_idx = load_split(args.split)
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    ds = NPZTextDataset(
        npz_path=args.npz,
        indices=test_idx,
        tokenizer_name=args.tokenizer,
        max_len=args.max_len,
        local_files_only=args.local_files_only,
    )
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_ulapm_from_artifacts(ckpt_path=args.ckpt, local_files_only=args.local_files_only)
    model.load_state_dict(load_state_dict_robust(args.ckpt), strict=True)
    if hasattr(model, "deterministic_latent"):
        model.deterministic_latent = True
    model.to(device)
    model.eval()

    y_c_true, y_c_pred = [], []
    va_true, va_pred = [], []
    b_true, b_pred = [], []

    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            c = batch["c"].cpu().numpy()
            va = batch["va"].cpu().numpy()
            b = batch["b"].cpu().numpy()

            out = model(input_ids, attn)
            c_pred = torch.argmax(out["c_logits"], dim=-1).cpu().numpy()
            va_hat = out["va_raw"].cpu().numpy()
            b_hat = out["b_probs"].cpu().numpy()

            y_c_true.extend(list(c))
            y_c_pred.extend(list(c_pred))
            va_true.append(va)
            va_pred.append(va_hat)
            b_true.append(b)
            b_pred.append(b_hat)

    va_true = np.concatenate(va_true, axis=0)
    va_pred = np.concatenate(va_pred, axis=0)
    b_true = np.concatenate(b_true, axis=0)
    b_pred = np.concatenate(b_pred, axis=0)

    metrics = {
        "emotion_acc": acc(y_c_true, y_c_pred),
        "emotion_macro_f1": macro_f1(y_c_true, y_c_pred, num_classes=7),
        "behavior_kl": behavior_kl(b_true, b_pred),
        "behavior_js": behavior_js(b_true, b_pred),
        "va_mse": mse(va_true, va_pred),
        "ccc_v": ccc(va_true[:, 0], va_pred[:, 0]),
        "ccc_a": ccc(va_true[:, 1], va_pred[:, 1]),
        "n_test": int(len(y_c_true)),
    }

    print("\n==================== OFFLINE TABLE I EVAL ====================")
    print(f"Emotion Acc      = {metrics['emotion_acc']:.4f}")
    print(f"Emotion Macro-F1 = {metrics['emotion_macro_f1']:.4f}")
    print(f"Behavior KL      = {metrics['behavior_kl']:.4f}")
    print(f"Behavior JS      = {metrics['behavior_js']:.4f}")
    print(f"VA MSE           = {metrics['va_mse']:.4f}")
    print(f"CCC(V)           = {metrics['ccc_v']:.4f}")
    print(f"CCC(A)           = {metrics['ccc_a']:.4f}")
    print("==============================================================")

    if args.out_json:
        payload = {
            "name": args.name,
            "ckpt": args.ckpt,
            "npz": args.npz,
            "split": args.split,
            "metrics": metrics,
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] wrote {args.out_json}")


if __name__ == "__main__":
    main()
