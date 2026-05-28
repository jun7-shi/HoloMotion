from __future__ import annotations

import torch


def _mean(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu().item())


def _p95(value: torch.Tensor) -> float:
    flat = value.detach().float().flatten().cpu()
    return float(torch.quantile(flat, 0.95).item())


def summarize_quiet_metrics(
    commanded_velocity: torch.Tensor,
    measured_velocity: torch.Tensor,
    touchdown_vz: torch.Tensor,
    peak_normal_force: torch.Tensor,
    force_rate: torch.Tensor,
    foot_slip: torch.Tensor,
    root_jerk: torch.Tensor,
) -> dict[str, float]:
    velocity_error = torch.linalg.norm(
        commanded_velocity - measured_velocity,
        dim=-1,
    )
    return {
        "velocity_error_mean": _mean(velocity_error),
        "touchdown_vz_mean": _mean(touchdown_vz),
        "peak_normal_force_p95": _p95(peak_normal_force),
        "force_rate_mean": _mean(force_rate),
        "foot_slip_mean": _mean(foot_slip),
        "root_jerk_mean": _mean(root_jerk),
    }
