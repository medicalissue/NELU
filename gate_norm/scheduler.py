"""(Deprecated) Warmup scheduler for the Gate Normalization γ.

γ is now a learnable parameter (``γ_raw`` with softplus reparam — see
:mod:`gate_norm.core`), so this scheduler is informational only: it
tracks the conceptual ramp curve for logging but no longer writes back
to the module. The module's own optimizer drives γ.

Kept for state-dict back-compat with older runs that did rely on the
ramp.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


def _gate_modules(model: nn.Module) -> Iterable[nn.Module]:
    for m in model.modules():
        if getattr(m, "_gate_norm_module", False) and hasattr(m, "gamma"):
            yield m


class GammaWarmup:
    """Drive γ from ``init`` to ``final`` over ``warmup_steps`` training steps.

    Parameters
    ----------
    model : nn.Module
        Any module containing :class:`gate_norm.core.GateNorm` instances.
        We scan the tree once at construction and cache the references.
    warmup_steps : int
        Number of training steps over which γ ramps. Must be ≥ 0.
        Setting it to 0 immediately fixes γ at ``final``.
    init : float, default 0.0
        Starting value of γ.
    final : float, default 1.0
        γ value held for the remainder of training, once warmup ends.
    schedule : {"linear", "cosine"}, default "linear"
        Linear matches ``torch.optim.lr_scheduler.LinearLR``; cosine
        matches the LR's cosine warmup (rare). Use "constant" to skip
        the ramp entirely (γ stays at ``final``).
    """

    def __init__(
        self,
        model: nn.Module,
        warmup_steps: int,
        *,
        init: float = 0.0,
        final: float = 1.0,
        schedule: str = "linear",
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if schedule not in {"linear", "cosine", "constant"}:
            raise ValueError(f"unknown schedule {schedule!r}")
        self.modules = list(_gate_modules(model))
        self.warmup_steps = int(warmup_steps)
        self.init = float(init)
        self.final = float(final)
        self.schedule = schedule
        self._step = 0
        # Apply initial value immediately so a forward before any step()
        # call still sees a sensible γ.
        self._set(init if (warmup_steps > 0 and schedule != "constant") else final)

    # ── core ──────────────────────────────────────────────────────────

    def gamma_at(self, step: int) -> float:
        """Pure function — return the γ value at training step ``step``."""
        if self.schedule == "constant" or self.warmup_steps == 0:
            return self.final
        if step >= self.warmup_steps:
            return self.final
        if step <= 0:
            return self.init
        frac = step / self.warmup_steps
        if self.schedule == "linear":
            return self.init + (self.final - self.init) * frac
        # cosine: 0 → final, smoother near the endpoints
        return self.init + 0.5 * (self.final - self.init) * (1.0 - math.cos(math.pi * frac))

    def step(self, step_idx: int | None = None) -> float:
        """Advance and apply. Returns the γ value just written."""
        if step_idx is None:
            self._step += 1
            step_idx = self._step
        else:
            self._step = int(step_idx)
        g = self.gamma_at(self._step)
        self._set(g)
        return g

    def _set(self, value: float) -> None:
        # γ is now a learnable parameter (γ_raw with softplus reparam),
        # so the warmup schedule is informational only — we do not write
        # back to the module. The module's own optimizer drives γ.
        return

    # ── checkpointing ─────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "step": self._step,
            "warmup_steps": self.warmup_steps,
            "init": self.init,
            "final": self.final,
            "schedule": self.schedule,
        }

    def load_state_dict(self, state: dict) -> None:
        self._step = int(state.get("step", 0))
        # warmup_steps / init / final / schedule are config-side; we trust
        # the construction args for the new run and only restore the
        # cursor so resume continues from the same γ value.
        self._set(self.gamma_at(self._step))

    # ── status ────────────────────────────────────────────────────────

    @property
    def current_gamma(self) -> float:
        return self.modules[0].gamma.item() if self.modules else float("nan")

    def __repr__(self) -> str:
        return (
            f"GammaWarmup(step={self._step}/{self.warmup_steps}, "
            f"γ={self.current_gamma:.4f}, schedule={self.schedule})"
        )
