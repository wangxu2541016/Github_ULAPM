# src/train_stage1_with_val_log.py
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.ulapm import ULAPMStage1


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(split_path: str):
    sp = np.load(split_path)
    # 兼容不同命名
    for k in ["train_idx", "tr_idx", "train"]:
        if k in sp: train_idx = sp[k]; break
    else: raise KeyError(f"split missing train idx keys, got {list(sp.keys())}")

    for k in ["val_idx", "valid_idx", "va_idx", "val", "valid"]:
        if k in sp: val_idx = sp[k]; break
    else: raise KeyError(f"split missing val idx keys, got {list(sp.keys())}")

    for k in ["test_idx", "te_idx", "test"]:
        if k in sp: test_idx = sp[k]; break
    else: raise KeyError(f"split missing test idx keys, got {list(sp.keys())}")

    return train_idx, val_idx, test_idx


class NpzDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.texts = data["texts"]
        self.c = data["c"]
        self.va = data["va"]
        self.b = data["b"]
        self.d = data["d"]
        self.sis = data["sis"]
        self.u = data["u"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            self.texts[idx],
            self.c[idx],
            self.va[idx],
            self.b[idx],
            self.d[idx],
            self.sis[idx],
            self.u[idx],
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
    texts, c, va, b, d, sis, u = zip(*batch)

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
    sis = torch.from_numpy(np.stack(sis).astype(np.float32))
    u = torch.from_numpy(np.stack(u).astype(np.float32))

    return enc["input_ids"], enc["attention_mask"], c, va, b, d, sis, u


def soft_ce_from_logits(logits, soft_targets):
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1).mean()


def weighted_sis_loss(pred_sis, target_sis, sis_weights, loss_type: str = "smooth_l1"):
    if loss_type == "smooth_l1":
        per_dim = F.smooth_l1_loss(pred_sis, target_sis, reduction="none")
    elif loss_type == "mse":
        per_dim = (pred_sis - target_sis) ** 2
    else:
        raise ValueError(f"Unsupported sis loss type: {loss_type}")
    return (per_dim * sis_weights.view(1, -1)).mean()


def compute_losses(out, c, va, b, d, sis, u, beta: float, sis_weights, sis_loss_type: str):
    Lc = F.cross_entropy(out["c_logits"], c)
    Lva = F.mse_loss(out["va_raw"], va)
    # b 是 soft 分布，正式实验统一用 soft CE，避免 Full/ablation 训练目标不一致
    Lb = soft_ce_from_logits(out["b_logits"], b)
    Ld = F.mse_loss(out["d"], d)
    Lsis = weighted_sis_loss(out["sis"], sis, sis_weights=sis_weights, loss_type=sis_loss_type)
    Lu = F.mse_loss(out["u"], u)

    Lkl = -0.5 * torch.mean(
        1 + out["log_sigma"] - out["mu"].pow(2) - out["log_sigma"].exp()
    )
    loss = Lc + Lva + Lb + Ld + Lsis + Lu + beta * Lkl

    return loss, {
        "Lc": Lc.detach().item(),
        "Lva": Lva.detach().item(),
        "Lb": Lb.detach().item(),
        "Ld": Ld.detach().item(),
        "Lsis": Lsis.detach().item(),
        "Lu": Lu.detach().item(),
        "Lkl": Lkl.detach().item(),
        "beta": float(beta),
    }


@torch.no_grad()
def eval_epoch(model, loader, device, beta: float, sis_weights, sis_loss_type: str):
    model.eval()
    loss_sum, n_sum = 0.0, 0
    for input_ids, attn, c, va, b, d, sis, u in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attn = attn.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        va = va.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        sis = sis.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)

        out = model(input_ids, attn)
        loss, _ = compute_losses(out, c, va, b, d, sis, u, beta=beta, sis_weights=sis_weights, sis_loss_type=sis_loss_type)

        bs = input_ids.size(0)
        loss_sum += loss.item() * bs
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
    ap.add_argument("--beta_scale", type=float, default=1.0,
                    help="Multiply the KL annealing factor by this value. Use 0.0 for zero-KL.")
    ap.add_argument("--log_every", type=int, default=20, help="log train step every N steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="runs/full")
    ap.add_argument("--deterministic_latent", action="store_true",
                    help="Use z=mu instead of sampling; combine with --beta_scale 0 for a deterministic zero-KL ablation.")
    ap.add_argument("--local_files_only", action="store_true",
                    help="Load tokenizer/model only from local Hugging Face cache.")
    ap.add_argument("--sis_head_type", default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--sis_hidden_dim", type=int, default=64)
    ap.add_argument("--sis_dropout", type=float, default=0.1)
    ap.add_argument("--sis_loss_type", default="smooth_l1", choices=["smooth_l1", "mse"])
    ap.add_argument("--sis_w_intent", type=float, default=1.0)
    ap.add_argument("--sis_w_engagement", type=float, default=4.0)
    ap.add_argument("--sis_w_closeness", type=float, default=1.0)
    ap.add_argument("--sis_w_risk", type=float, default=2.0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)
    base_ds = NpzDataset(args.npz)
    train_idx, val_idx, test_idx = load_split(args.split)

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

    model = ULAPMStage1(
        encoder_name=args.tokenizer,
        deterministic_latent=args.deterministic_latent,
        sis_head_type=args.sis_head_type,
        sis_hidden_dim=args.sis_hidden_dim,
        sis_dropout=args.sis_dropout,
        local_files_only=args.local_files_only,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sis_weights = torch.tensor(
        [args.sis_w_intent, args.sis_w_engagement, args.sis_w_closeness, args.sis_w_risk],
        dtype=torch.float32,
        device=device,
    )

    # ===== CSV (step-level train + epoch-level val) =====
    csv_path = os.path.join(args.out_dir, "logs", "train_val_steps.csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["global_step", "epoch", "split", "loss", "Lc", "Lva", "Lb", "Ld", "Lsis", "Lu", "Lkl", "beta", "seconds"])
    f.flush()

    best_val = 1e18
    global_step = 0

    try:
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()

            beta = args.beta_scale * min(1.0, (epoch + 1) / max(1, args.kl_anneal_epochs))
            pbar = tqdm(train_loader, desc=f"train epoch {epoch+1}/{args.epochs}", ncols=95)

            for batch in pbar:
                input_ids, attn, c, va, b, d, sis, u = batch
                input_ids = input_ids.to(device, non_blocking=True)
                attn = attn.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                va = va.to(device, non_blocking=True)
                b = b.to(device, non_blocking=True)
                d = d.to(device, non_blocking=True)
                sis = sis.to(device, non_blocking=True)
                u = u.to(device, non_blocking=True)

                out = model(input_ids, attn)
                loss, parts = compute_losses(
                    out, c, va, b, d, sis, u,
                    beta=beta,
                    sis_weights=sis_weights,
                    sis_loss_type=args.sis_loss_type,
                )

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                global_step += 1

                # tqdm显示
                pbar.set_postfix(loss=float(loss.detach().item()), beta=float(beta), Lu=float(parts["Lu"]), Lsis=float(parts["Lsis"]))

                # step log（训练曲线用这个）
                if global_step % args.log_every == 0:
                    sec = time.time() - t0
                    w.writerow([global_step, epoch, "train",
                                f"{loss.detach().item():.6f}",
                                f"{parts['Lc']:.6f}", f"{parts['Lva']:.6f}", f"{parts['Lb']:.6f}",
                                f"{parts['Ld']:.6f}", f"{parts['Lsis']:.6f}", f"{parts['Lu']:.6f}",
                                f"{parts['Lkl']:.6f}", f"{parts['beta']:.4f}", f"{sec:.2f}"])
                    f.flush()

            # epoch结束：val
            val_loss = eval_epoch(
                model,
                val_loader,
                device,
                beta=beta,
                sis_weights=sis_weights,
                sis_loss_type=args.sis_loss_type,
            )
            sec = time.time() - t0
            w.writerow([global_step, epoch, "val",
                        f"{val_loss:.6f}", "", "", "", "", "", "", "", f"{beta:.4f}", f"{sec:.2f}"])
            f.flush()
            print(f"[VAL] epoch={epoch+1} beta={beta:.3f} val_loss={val_loss:.6f}")

            # save best
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = os.path.join(args.out_dir, "stage1_best.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"[SAVE] best ckpt saved: {ckpt_path} (val_loss={best_val:.6f})")

        # final
        final_path = os.path.join(args.out_dir, "stage1_last.pt")
        torch.save(model.state_dict(), final_path)
        print(f"[SAVE] last ckpt saved: {final_path}")

    finally:
        f.close()
        print("saved log:", csv_path)


if __name__ == "__main__":
    main()
