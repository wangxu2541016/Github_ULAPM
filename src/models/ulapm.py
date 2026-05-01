import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.models.constraints import HardConstraintHead


MODEL_CONFIG_KEYS = {
    "encoder_name",
    "adapter_dim",
    "z_dim",
    "num_emotions",
    "num_behaviors",
    "deterministic_latent",
    "sis_head_type",
    "sis_hidden_dim",
    "sis_dropout",
}


def model_kwargs_from_train_config(config: dict | None) -> dict:
    config = config or {}
    kwargs = {}
    for key in MODEL_CONFIG_KEYS:
        if key in config:
            kwargs[key] = config[key]
    if "encoder_name" not in kwargs and "tokenizer" in config:
        kwargs["encoder_name"] = config["tokenizer"]
    return kwargs


def load_model_kwargs_from_artifacts(ckpt_path: str | Path | None = None, train_config_path: str | Path | None = None) -> dict:
    candidate = None
    if train_config_path:
        candidate = Path(train_config_path)
    elif ckpt_path:
        candidate = Path(ckpt_path).resolve().with_name("train_config.json")
    if not candidate or not candidate.exists():
        return {}
    try:
        return model_kwargs_from_train_config(json.loads(candidate.read_text(encoding="utf-8")))
    except Exception:
        return {}


def build_ulapm_from_artifacts(ckpt_path: str | Path | None = None, train_config_path: str | Path | None = None, **overrides):
    kwargs = load_model_kwargs_from_artifacts(ckpt_path=ckpt_path, train_config_path=train_config_path)
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return ULAPMStage1(**kwargs)


class SimpleAdapter(nn.Module):
    """A tiny adapter layer to keep it light (toy/prototype)."""
    def __init__(self, hidden_size: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        # residual adapter
        return x + self.up(self.act(self.down(x)))


class SISMLPHead(nn.Module):
    """A slightly stronger SIS head that keeps per-dimension outputs independent."""
    def __init__(self, z_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.intent = nn.Linear(hidden_dim, 1)
        self.engagement = nn.Linear(hidden_dim, 1)
        self.closeness = nn.Linear(hidden_dim, 1)
        self.risk = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        h = self.trunk(z)
        return torch.cat(
            [
                self.intent(h),
                self.engagement(h),
                self.closeness(h),
                self.risk(h),
            ],
            dim=1,
        )


class ULAPMStage1(nn.Module):
    """
    Stage1 prototype:
      text -> encoder -> h -> latent z ~ q_phi(z|x)
      z -> heads -> emotion c, VA, behavior b, distance d, SIS, action u
      then HardConstraintHead enforces valid ranges for d/sis/u.

    Returns:
      - raw outputs (for losses if needed)
      - constrained outputs (for inference + safe training)
    """
    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        adapter_dim: int = 64,
        z_dim: int = 128,
        num_emotions: int = 7,
        num_behaviors: int = 5,
        deterministic_latent: bool = False,
        sis_head_type: str = "linear",
        sis_hidden_dim: int = 64,
        sis_dropout: float = 0.1,
        local_files_only: bool = False,
        hard_cfg: dict | None = None,
    ):
        super().__init__()
        self.model_config = {
            "encoder_name": encoder_name,
            "adapter_dim": int(adapter_dim),
            "z_dim": int(z_dim),
            "num_emotions": int(num_emotions),
            "num_behaviors": int(num_behaviors),
            "deterministic_latent": bool(deterministic_latent),
            "sis_head_type": sis_head_type,
            "sis_hidden_dim": int(sis_hidden_dim),
            "sis_dropout": float(sis_dropout),
        }
        self.deterministic_latent = bool(deterministic_latent)

        # -------- text encoder --------
        self.encoder = AutoModel.from_pretrained(encoder_name, local_files_only=local_files_only)
        hidden = self.encoder.config.hidden_size
        self.adapter = SimpleAdapter(hidden, bottleneck=adapter_dim)

        # -------- q_phi(z|x): produce mu, log_sigma --------
        self.mu = nn.Linear(hidden, z_dim)
        self.log_sigma = nn.Linear(hidden, z_dim)

        # -------- heads from z --------
        self.head_c = nn.Linear(z_dim, num_emotions)   # logits
        self.head_va = nn.Linear(z_dim, 2)            # (v,a)
        self.head_b = nn.Linear(z_dim, num_behaviors) # logits/probs
        self.head_d = nn.Linear(z_dim, 1)             # raw distance
        if sis_head_type == "linear":
            self.head_sis = nn.Linear(z_dim, 4)       # raw SIS [I,E,P,R]
        elif sis_head_type == "mlp":
            self.head_sis = SISMLPHead(z_dim, hidden_dim=sis_hidden_dim, dropout=sis_dropout)
        else:
            raise ValueError(f"Unsupported sis_head_type={sis_head_type!r}")
        self.head_u = nn.Linear(z_dim, 3)             # raw u [Δr, v_r, τ]

        # -------- hard constraints head --------
        if hard_cfg is None:
            hard_cfg = dict(d_min=0.3, d_max=2.5, dr_max=0.6, vmin=0.05, vmax=0.30, tmin=0.5, tmax=3.0)
        self.hard = HardConstraintHead(**hard_cfg)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT: use first token as CLS-like
        h_cls = out.last_hidden_state[:, 0, :]  # (B,H)
        h_cls = self.adapter(h_cls)
        return h_cls

    @staticmethod
    def reparameterize(mu, log_sigma):
        # log_sigma is log(std); std = exp(log_sigma)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(log_sigma) * eps
        return z

    def forward(self, input_ids, attention_mask):
        # 1) text -> h
        h = self.encode(input_ids, attention_mask)

        # 2) q_phi(z|x): mu/log_sigma -> z
        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        if self.deterministic_latent:
            z = mu
        else:
            z = self.reparameterize(mu, log_sigma)

        # 3) raw heads
        c_logits = self.head_c(z)
        va_raw = self.head_va(z)
        b_logits = self.head_b(z)
        d_raw = self.head_d(z)
        sis_raw = self.head_sis(z)
        u_raw = self.head_u(z)

        # 4) apply hard constraints for safe outputs
        d_con, sis_con, u_con = self.hard(d_raw, sis_raw, u_raw)

        # 5) behavior probs (softmax)
        b_probs = F.softmax(b_logits, dim=-1)

        return {
            # latent
            "z": z,
            "mu": mu,
            "log_sigma": log_sigma,

            # classification/regression heads
            "c_logits": c_logits,
            "va_raw": va_raw,
            "b_logits": b_logits,
            "b_probs": b_probs,

            # raw (unconstrained)
            "d_raw": d_raw,
            "sis_raw": sis_raw,
            "u_raw": u_raw,

            # constrained (what you should print/use in run_once)
            "d": d_con,
            "sis": sis_con,
            "u": u_con,
        }
