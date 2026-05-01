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

from src.models.plain_multitask import PlainMultiTask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(split_path):
    sp = np.load(split_path, allow_pickle=True)
    if {"train_idx", "val_idx", "test_idx"}.issubset(sp.files):
        return sp["train_idx"], sp["val_idx"], sp["test_idx"]
    if {"train", "val", "test"}.issubset(sp.files):
        return sp["train"], sp["val"], sp["test"]
    raise KeyError(f"Split file keys not recognized. got={sp.files}")


class NpzDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.texts = data["texts"]
        self.c = data["c"].astype(np.int64)
        self.va = data["va"].astype(np.float32)
        self.b = data["b"].astype(np.float32)
        self.d = data["d"].astype(np.float32)
        self.u = data["u"].astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.c[idx], self.va[idx], self.b[idx], self.d[idx], self.u[idx]


class SubsetByIndex(Dataset):
    def __init__(self, base_ds, indices):
        self.base = base_ds
        self.indices = indices.astype(np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]


def collate_fn(batch, tokenizer, max_len):
    texts, c, va, b, d, u = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return (
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(c, dtype=torch.long),
        torch.from_numpy(np.stack(va).astype(np.float32)),
        torch.from_numpy(np.stack(b).astype(np.float32)),
        torch.from_numpy(np.stack(d).astype(np.float32)),
        torch.from_numpy(np.stack(u).astype(np.float32)),
    )


def soft_ce_from_logits(logits, soft_targets):
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1).mean()


def compute_losses(out, c, va, b, d, u):
    l_c = F.cross_entropy(out["c_logits"], c)
    l_va = F.mse_loss(out["va_raw"], va)
    l_b = soft_ce_from_logits(out["b_logits"], b)
    l_d = F.mse_loss(out["d"], d)
    l_u = F.mse_loss(out["u"], u)
    loss = l_c + l_va + l_b + l_d + l_u
    return loss, {
        "Lc": float(l_c.detach().cpu()),
        "Lva": float(l_va.detach().cpu()),
        "Lb": float(l_b.detach().cpu()),
        "Ld": float(l_d.detach().cpu()),
        "Lu": float(l_u.detach().cpu()),
    }


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    loss_sum, n_sum = 0.0, 0
    for input_ids, attn, c, va, b, d, u in loader:
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        c = c.to(device)
        va = va.to(device)
        b = b.to(device)
        d = d.to(device)
        u = u.to(device)
        out = model(input_ids, attn)
        loss, _ = compute_losses(out, c, va, b, d, u)
        bs = input_ids.size(0)
        loss_sum += float(loss.detach().cpu()) * bs
        n_sum += bs
    return loss_sum / max(1, n_sum)


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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="runs/b2_plain")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

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
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_len),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_len),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = PlainMultiTask(encoder_name=args.tokenizer).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    csv_path = os.path.join(args.out_dir, "logs", "train_val_steps.csv")
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["global_step", "epoch", "split", "loss", "Lc", "Lva", "Lb", "Ld", "Lu", "seconds"])
    f.flush()

    best_val = 1e18
    global_step = 0
    try:
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            pbar = tqdm(train_loader, desc=f"train B2 {epoch + 1}/{args.epochs}", ncols=95)
            for batch in pbar:
                input_ids, attn, c, va, b, d, u = batch
                input_ids = input_ids.to(device, non_blocking=True)
                attn = attn.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                va = va.to(device, non_blocking=True)
                b = b.to(device, non_blocking=True)
                d = d.to(device, non_blocking=True)
                u = u.to(device, non_blocking=True)

                out = model(input_ids, attn)
                loss, parts = compute_losses(out, c, va, b, d, u)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                global_step += 1
                pbar.set_postfix(loss=float(loss.detach().cpu()), Ld=parts["Ld"], Lu=parts["Lu"])
                if global_step % 20 == 0:
                    writer.writerow([
                        global_step,
                        epoch,
                        "train",
                        f"{float(loss.detach().cpu()):.6f}",
                        f"{parts['Lc']:.6f}",
                        f"{parts['Lva']:.6f}",
                        f"{parts['Lb']:.6f}",
                        f"{parts['Ld']:.6f}",
                        f"{parts['Lu']:.6f}",
                        f"{time.time() - t0:.2f}",
                    ])
                    f.flush()

            val_loss = eval_epoch(model, val_loader, device)
            writer.writerow([global_step, epoch, "val", f"{val_loss:.6f}", "", "", "", "", "", f"{time.time() - t0:.2f}"])
            f.flush()
            print(f"[VAL] epoch={epoch + 1} val_loss={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = os.path.join(args.out_dir, "b2_plain_best.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"[SAVE] best ckpt saved: {ckpt_path} (val_loss={best_val:.6f})")

        last_path = os.path.join(args.out_dir, "b2_plain_last.pt")
        torch.save(model.state_dict(), last_path)
        print(f"[SAVE] last ckpt saved: {last_path}")
    finally:
        f.close()
        print("saved log:", csv_path)


if __name__ == "__main__":
    main()
