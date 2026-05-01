import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PlainMultiTask(nn.Module):
    """
    B2 baseline:
      text encoder -> shared representation -> direct prediction heads.

    No VAE bottleneck, no SIS head, and no hard constraints. This baseline
    tests whether ordinary multi-task learning is enough for behavior-distance
    planning.
    """

    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        hidden_dim: int = 256,
        num_emotions: int = 7,
        num_behaviors: int = 5,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, local_files_only=local_files_only)
        hidden = self.encoder.config.hidden_size

        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.head_c = nn.Linear(hidden_dim, num_emotions)
        self.head_va = nn.Linear(hidden_dim, 2)
        self.head_b = nn.Linear(hidden_dim, num_behaviors)
        self.head_d = nn.Linear(hidden_dim, 1)
        self.head_u = nn.Linear(hidden_dim, 3)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask):
        h = self.shared(self.encode(input_ids, attention_mask))
        b_logits = self.head_b(h)
        d_raw = self.head_d(h)
        u_raw = self.head_u(h)
        return {
            "c_logits": self.head_c(h),
            "va_raw": self.head_va(h),
            "b_logits": b_logits,
            "b_probs": F.softmax(b_logits, dim=-1),
            "d_raw": d_raw,
            "d": d_raw,
            "u_raw": u_raw,
            "u": u_raw,
        }
