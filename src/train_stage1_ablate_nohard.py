import os
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.ulapm import ULAPMStage1


# -----------------------------
# Dataset
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NPZDataset(Dataset):
    """
    Expects npz keys:
      texts (object array of str)
      c     (int64)   emotion class (0..6)
      va    (float32) [v,a]
      b     (float32) behavior probs (5,)
      d     (float32) [dist]
      sis   (float32) [I,E,P,R]
      u     (float32) [dr, vr, tau]
    """
    def __init__(self, npz_path="data/processed/ulapm_behavior_sis_u_toy.npz", max_len=64, bert_name="distilbert-base-uncased"):
        data = np.load(npz_path, allow_pickle=True)
        self.texts = data["texts"]
        self.c = data["c"].astype(np.int64)
        self.va = data["va"].astype(np.float32)
        self.b = data["b"].astype(np.float32)
        self.d = data["d"].astype(np.float32)
        self.sis = data["sis"].astype(np.float32)
        self.u = data["u"].astype(np.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "c": torch.tensor(self.c[idx], dtype=torch.long),
            "va": torch.tensor(self.va[idx], dtype=torch.float32),
            "b": torch.tensor(self.b[idx], dtype=torch.float32),
            "d": torch.tensor(self.d[idx], dtype=torch.float32),      # shape (1,)
            "sis": torch.tensor(self.sis[idx], dtype=torch.float32),  # shape (4,)
            "u": torch.tensor(self.u[idx], dtype=torch.float32),      # shape (3,)
        }
        return item


def make_split(n, out_path, seed=0, train_ratio=0.8, val_ratio=0.1):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, train=train_idx, val=val_idx, test=test_idx)
    return train_idx, val_idx, test_idx


def load_split(split_path):
    sp = np.load(split_path)
    files = set(sp.files)
    if {"train_idx", "val_idx", "test_idx"}.issubset(files):
        return sp["train_idx"], sp["val_idx"], sp["test_idx"]
    if {"train", "val", "test"}.issubset(files):
        return sp["train"], sp["val"], sp["test"]
    raise KeyError(f"Split file keys not recognized. got={sp.files}")


# -----------------------------
# Loss helpers
# -----------------------------
def soft_ce_from_logits(logits, soft_targets):
    """
    logits: (B, K)
    soft_targets: (B, K), sum=1
    """
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1).mean()


def kl_normal_standard(mu, log_sigma):
    """
    KL(q(z|x)=N(mu, sigma) || N(0,1))
    log_sigma follows ULAPMStage1.reparameterize: std = exp(log_sigma).
    """
    return 0.5 * torch.mean(torch.sum(mu * mu + torch.exp(2 * log_sigma) - 1.0 - 2 * log_sigma, dim=-1))


def weighted_sis_loss(pred_sis, target_sis, sis_weights):
    per_dim = F.smooth_l1_loss(pred_sis, target_sis, reduction="none")
    return (per_dim * sis_weights.view(1, -1)).mean()


# -----------------------------
# Train
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="data/processed/ulapm_behavior_sis_u_toy.npz")
    ap.add_argument("--split", type=str, default="data/processed/split_toy_sis_u.npz")
    ap.add_argument("--bert", type=str, default="distilbert-base-uncased")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="stage1_ablate_nohard.pt")
    ap.add_argument("--sis_head_type", default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--sis_hidden_dim", type=int, default=64)
    ap.add_argument("--sis_dropout", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = NPZDataset(npz_path=args.npz, max_len=args.max_len, bert_name=args.bert)
    n = len(ds)

    if os.path.exists(args.split):
        train_idx, val_idx, test_idx = load_split(args.split)
    else:
        train_idx, val_idx, test_idx = make_split(n, args.split, seed=args.seed)

    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=args.batch, shuffle=False)

    model = ULAPMStage1(
        encoder_name=args.bert,
        sis_head_type=args.sis_head_type,
        sis_hidden_dim=args.sis_hidden_dim,
        sis_dropout=args.sis_dropout,
    )
    model.to(device)
    sis_weights = torch.tensor([1.0, 4.0, 1.0, 2.0], dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    save_path = Path(args.save).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = save_path.with_name("train_config.json")
    config_path.write_text(
        json.dumps(
            {
                "npz": args.npz,
                "split": args.split,
                "tokenizer": args.bert,
                "bert": args.bert,
                "max_len": args.max_len,
                "batch": args.batch,
                "epochs": args.epochs,
                "lr": args.lr,
                "seed": args.seed,
                "save": args.save,
                "sis_head_type": args.sis_head_type,
                "sis_hidden_dim": args.sis_hidden_dim,
                "sis_dropout": args.sis_dropout,
                "sis_w_intent": 1.0,
                "sis_w_engagement": 4.0,
                "sis_w_closeness": 1.0,
                "sis_w_risk": 2.0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # beta schedule: epoch0=0, epoch>=1=1 （跟你之前打印 beta 类似）
    def beta_of_epoch(ep):
        return 0.0 if ep == 0 else 1.0

    for ep in range(args.epochs):
        model.train()
        beta = beta_of_epoch(ep)

        pbar = tqdm(train_loader, desc=f"epoch {ep}", ncols=110)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            c = batch["c"].to(device)
            va = batch["va"].to(device)          # (B,2)
            b = batch["b"].to(device)            # (B,5) soft labels
            d = batch["d"].to(device)            # (B,1)
            sis = batch["sis"].to(device)        # (B,4)
            u = batch["u"].to(device)            # (B,3)

            out = model(input_ids, attn)

            # 1) emotion
            # 你的模型 keys 是 c_logits / b_logits / va_raw / d_raw / sis_raw / u_raw
            Lc = F.cross_entropy(out["c_logits"], c)

            # 2) VA (reg)
            Lva = F.mse_loss(out["va_raw"], va)

            # 3) behavior soft labels
            Lb = soft_ce_from_logits(out["b_logits"], b)

            # 4) distance / sis / u
            # 关键：这里用 RAW（不经过 HardConstraintHead）=> nohard 消融
            Ld = F.l1_loss(out["d_raw"], d)
            Lsis = weighted_sis_loss(out["sis_raw"], sis, sis_weights=sis_weights)
            Lu = F.mse_loss(out["u_raw"], u)

            # 5) KL
            Lkl = kl_normal_standard(out["mu"], out["log_sigma"])

            loss = Lc + Lva + Lb + Ld + Lsis + Lu + beta * Lkl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(
                loss=float(loss.detach().cpu()),
                Lsis=float(Lsis.detach().cpu()),
                Lu=float(Lu.detach().cpu()),
                beta=float(beta),
            )

        # quick val (optional)
        model.eval()
        with torch.no_grad():
            tot = 0
            lsum = 0.0
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                c = batch["c"].to(device)
                va = batch["va"].to(device)
                b = batch["b"].to(device)
                d = batch["d"].to(device)
                sis = batch["sis"].to(device)
                u = batch["u"].to(device)

                out = model(input_ids, attn)
                Lc = F.cross_entropy(out["c_logits"], c)
                Lva = F.mse_loss(out["va_raw"], va)
                Lb = soft_ce_from_logits(out["b_logits"], b)
                Ld = F.l1_loss(out["d_raw"], d)
                Lsis = weighted_sis_loss(out["sis_raw"], sis, sis_weights=sis_weights)
                Lu = F.mse_loss(out["u_raw"], u)
                Lkl = kl_normal_standard(out["mu"], out["log_sigma"])
                beta = beta_of_epoch(ep)
                loss = Lc + Lva + Lb + Ld + Lsis + Lu + beta * Lkl
                lsum += float(loss.cpu()) * input_ids.size(0)
                tot += input_ids.size(0)
            if tot > 0:
                print(f"[val] epoch={ep} loss={lsum/tot:.4f}")

    torch.save(model.state_dict(), args.save)
    print("saved:", args.save)


if __name__ == "__main__":
    main()
