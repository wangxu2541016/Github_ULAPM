import torch
import torch.nn as nn


class HardConstraintHead(nn.Module):
    """
    Hard constraints to guarantee physically/socially valid outputs.

    Inputs:
      pred_d_raw   : (B, 1) raw distance
      pred_sis_raw : (B, 4) raw SIS [I, E, P, R]
      pred_u_raw   : (B, 3) raw action u [Δr, v_r, τ]

    Outputs (all constrained):
      pred_d   : (B, 1) in [d_min, d_max]
      pred_sis : (B, 4) I raw (kept), E/P/R in [0, 1]
      pred_u   : (B, 3) Δr in [-dr_max, dr_max], v_r in [vmin, vmax], τ in [tmin, tmax]
    """
    def __init__(
        self,
        d_min=0.3,
        d_max=2.5,
        dr_max=0.6,
        vmin=0.05,
        vmax=0.30,
        tmin=0.5,
        tmax=3.0,
    ):
        super().__init__()
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.dr_max = float(dr_max)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.tmin = float(tmin)
        self.tmax = float(tmax)

    def forward(self, pred_d_raw, pred_sis_raw, pred_u_raw):
        # ---------- distance d: map to [d_min, d_max] ----------
        # sigmoid ensures bounded + smooth gradients
        pred_d = self.d_min + (self.d_max - self.d_min) * torch.sigmoid(pred_d_raw)

        # ---------- SIS: E/P/R -> [0,1], keep I raw ----------
        # pred_sis_raw: (B,4) => [I, E, P, R]
        I = pred_sis_raw[:, 0:1]                 # keep as continuous raw
        EPR = torch.sigmoid(pred_sis_raw[:, 1:4])  # force to [0,1]
        pred_sis = torch.cat([I, EPR], dim=1)

        # ---------- action u = [Δr, v_r, τ] ----------
        # Δr in [-dr_max, dr_max] via tanh
        dr = self.dr_max * torch.tanh(pred_u_raw[:, 0:1])

        # v_r in [vmin, vmax] via sigmoid
        vr = self.vmin + (self.vmax - self.vmin) * torch.sigmoid(pred_u_raw[:, 1:2])

        # τ in [tmin, tmax] via sigmoid
        tau = self.tmin + (self.tmax - self.tmin) * torch.sigmoid(pred_u_raw[:, 2:3])

        pred_u = torch.cat([dr, vr, tau], dim=1)

        return pred_d, pred_sis, pred_u

