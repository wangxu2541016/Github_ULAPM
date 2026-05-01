import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.ulapm import ULAPMStage1


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(split_path: str):
    sp = np.load(split_path, allow_pickle=True)
    files = set(sp.files)
    if {"train_idx", "val_idx", "test_idx"}.issubset(files):
        return sp["train_idx"], sp["val_idx"], sp["test_idx"]
    if {"train", "val", "test"}.issubset(files):
        return sp["train"], sp["val"], sp["test"]
    raise KeyError(f"Split file keys not recognized. got={sp.files}")


class NpzDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.texts = data["texts"]
        self.c = data["c"]
        self.va = data["va"]
        self.b = data["b"]
        self.d = data["d"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            self.texts[idx],
            self.c[idx],
            self.va[idx],
            self.b[idx],
            self.d[idx],
        )


class SubsetByIndex(Dataset):
    def __init__(self, base_ds: Dataset, indices: np.ndarray):
        self.base = base_ds
        self.idx = indices.astype(np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[int(self.idx[i])]


def collate_fn(batch, tokenizer, max_len=128):
    texts, c, va, b, d = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    c = torch.tensor(c, dtype=torch.long)
    va = torch.from_numpy(np.stack(va).astype(np.float32))
    b = torch.from_numpy(np.stack(b).astype(np.float32))
    d = torch.from_numpy(np.stack(d).astype(np.float32))
    return enc["input_ids"], enc["attention_mask"], c, va, b, d


def soft_ce_from_logits(logits, soft_targets):
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1).mean()


def compute_losses(out, c, va, b, d, beta: float):
    lc = F.cross_entropy(out["c_logits"], c)
    lva = F.mse_loss(out["va_raw"], va)
    lb = soft_ce_from_logits(out["b_logits"], b)
    ld = F.mse_loss(out["d"], d)
    lkl = -0.5 * torch.mean(
        1 + out["log_sigma"] - out["mu"].pow(2) - out["log_sigma"].exp()
    )
    loss = lc + lva + lb + ld + beta * lkl
    return loss, {
        "Lc": float(lc.detach().cpu()),
        "Lva": float(lva.detach().cpu()),
        "Lb": float(lb.detach().cpu()),
        "Ld": float(ld.detach().cpu()),
        "Lkl": float(lkl.detach().cpu()),
        "beta": float(beta),
    }


@torch.no_grad()
def eval_epoch(model, loader, device, beta: float):
    model.eval()
    loss_sum, n_sum = 0.0, 0
    for input_ids, attn, c, va, b, d in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attn = attn.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        va = va.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        out = model(input_ids, attn)
        loss, _ = compute_losses(out, c, va, b, d, beta=beta)
        bs = input_ids.size(0)
        loss_sum += float(loss.detach().cpu()) * bs
        n_sum += bs
    return loss_sum / max(n_sum, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/processed/ulapm_behavior_sis_u_toy.npz")
    ap.add_argument("--split", default="data/processed/split_toy_sis_u.npz")
    ap.add_argument("--tokenizer", default="distilbert-base-uncased")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl_anneal_epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="runs/nosis_clean")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    base_ds = NpzDataset(args.npz)
    train_idx, val_idx, _ = load_split(args.split)

    train_ds = SubsetByIndex(base_ds, train_idx)
    val_ds = SubsetByIndex(base_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, max_len=args.max_len),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, max_len=args.max_len),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ULAPMStage1().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    csv_path = os.path.join(args.out_dir, "logs", "train_val_steps.csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["global_step", "epoch", "split", "loss", "Lc", "Lva", "Lb", "Ld", "Lkl", "beta", "seconds"])
    f.flush()

    best_val = 1e18
    global_step = 0

    try:
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            beta = min(1.0, (epoch + 1) / max(1, args.kl_anneal_epochs))
            pbar = tqdm(train_loader, desc=f"train NoSIS {epoch + 1}/{args.epochs}", ncols=95)

            for batch in pbar:
                input_ids, attn, c, va, b, d = batch
                input_ids = input_ids.to(device, non_blocking=True)
                attn = attn.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                va = va.to(device, non_blocking=True)
                b = b.to(device, non_blocking=True)
                d = d.to(device, non_blocking=True)

                out = model(input_ids, attn)
                loss, parts = compute_losses(out, c, va, b, d, beta=beta)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                global_step += 1
                pbar.set_postfix(loss=float(loss.detach().cpu()), Ld=parts["Ld"], beta=float(beta))

                if global_step % 20 == 0:
                    w.writerow([
                        global_step,
                        epoch,
                        "train",
                        f"{float(loss.detach().cpu()):.6f}",
                        f"{parts['Lc']:.6f}",
                        f"{parts['Lva']:.6f}",
                        f"{parts['Lb']:.6f}",
                        f"{parts['Ld']:.6f}",
                        f"{parts['Lkl']:.6f}",
                        f"{parts['beta']:.4f}",
                        f"{time.time() - t0:.2f}",
                    ])
                    f.flush()

            val_loss = eval_epoch(model, val_loader, device, beta=beta)
            w.writerow([global_step, epoch, "val", f"{val_loss:.6f}", "", "", "", "", "", f"{beta:.4f}", f"{time.time() - t0:.2f}"])
            f.flush()
            print(f"[VAL] epoch={epoch + 1} beta={beta:.3f} val_loss={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = os.path.join(args.out_dir, "stage1_nosis_best.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"[SAVE] best ckpt saved: {ckpt_path} (val_loss={best_val:.6f})")

        final_path = os.path.join(args.out_dir, "stage1_nosis_last.pt")
        torch.save(model.state_dict(), final_path)
        print(f"[SAVE] last ckpt saved: {final_path}")
    finally:
        f.close()
        print("saved log:", csv_path)


if __name__ == "__main__":
    main()
