from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class ControllerState:
    workload: str
    current_ci: float
    forecast_ci: float
    queue_len: int
    prev_mode: str | None
    tenant_id: str
    rolling_carbon_g: float
    slo_ms: float


class Controller(Protocol):
    name: str

    def choose_mode(self, state: ControllerState) -> str:
        ...


@dataclass(frozen=True)
class ModeSummary:
    mode_id: str
    precision: str
    capacity_k: int
    latency_p95_ms: float
    energy_mean_Wh: float
    accuracy: float


class StaticHQ:
    name = "static_hq"

    def __init__(self, modes: list[ModeSummary]):
        self._hq = sorted(modes, key=lambda m: (m.accuracy, -m.energy_mean_Wh), reverse=True)[0].mode_id

    def choose_mode(self, state: ControllerState) -> str:
        return self._hq


class StaticEco:
    name = "static_eco"

    def __init__(self, modes: list[ModeSummary]):
        self._eco = sorted(modes, key=lambda m: (m.energy_mean_Wh, m.latency_p95_ms))[0].mode_id

    def choose_mode(self, state: ControllerState) -> str:
        return self._eco


class ReactivePrecision:
    name = "reactive_precision"

    def __init__(self, modes: list[ModeSummary], t1: float = 180.0, t2: float = 350.0):
        max_k = max(m.capacity_k for m in modes)
        self._cands = sorted(
            [m for m in modes if m.capacity_k == max_k],
            key=lambda m: ({"fp16": 0, "int8": 1, "int4": 2}.get(m.precision, 9), m.energy_mean_Wh),
        )
        if len(self._cands) == 0:
            raise ValueError("ReactivePrecision needs at least one mode")
        self._t1 = t1
        self._t2 = t2

    def choose_mode(self, state: ControllerState) -> str:
        if len(self._cands) == 1:
            return self._cands[0].mode_id
        if state.current_ci <= self._t1:
            return self._cands[0].mode_id
        if state.current_ci <= self._t2:
            return self._cands[min(1, len(self._cands) - 1)].mode_id
        return self._cands[-1].mode_id


class ReactiveJoint:
    name = "reactive_joint"

    def __init__(self, modes: list[ModeSummary], t1: float = 180.0, t2: float = 350.0, hysteresis: float = 20.0):
        self._modes = sorted(modes, key=lambda m: (m.energy_mean_Wh, m.latency_p95_ms))
        self._t1 = t1
        self._t2 = t2
        self._h = hysteresis
        self._last_bucket: int | None = None

    def _bucket(self, ci: float) -> int:
        if ci <= self._t1:
            return 0
        if ci <= self._t2:
            return 1
        return 2

    def choose_mode(self, state: ControllerState) -> str:
        b = self._bucket(state.current_ci)
        if self._last_bucket is not None and b != self._last_bucket:
            if abs(state.current_ci - (self._t1 if b == 0 else self._t2)) < self._h:
                b = self._last_bucket

        self._last_bucket = b
        idx = min(b, len(self._modes) - 1)
        return self._modes[idx].mode_id


class ForecastBudgetedJoint:
    name = "forecast_budgeted_joint"

    def __init__(
        self,
        modes: list[ModeSummary],
        lambda_slo: float = 5.0,
        mu_switch: float = 0.5,
        nu_budget: float = 1.0,
        alpha_ci: float = 0.5,
        target_rolling_carbon_g: float = 5000.0,
    ):
        self._modes = modes
        self._lambda = lambda_slo
        self._mu = mu_switch
        self._nu = nu_budget
        self._alpha_ci = alpha_ci
        self._target = target_rolling_carbon_g

    def choose_mode(self, state: ControllerState) -> str:
        ci_tilde = self._alpha_ci * state.current_ci + (1.0 - self._alpha_ci) * state.forecast_ci

        best = None
        best_score = None
        for m in self._modes:
            est_carbon_g = m.energy_mean_Wh / 1000.0 * ci_tilde
            est_tail = m.latency_p95_ms * max(1, state.queue_len + 1)
            slo_risk = max(0.0, est_tail - state.slo_ms) / max(state.slo_ms, 1e-9)
            switch_cost = 1.0 if (state.prev_mode is not None and state.prev_mode != m.mode_id) else 0.0
            budget_penalty = max(0.0, state.rolling_carbon_g - self._target) / max(self._target, 1e-9)
            score = est_carbon_g + self._lambda * slo_risk + self._mu * switch_cost + self._nu * budget_penalty
            if best_score is None or score < best_score:
                best_score = score
                best = m.mode_id

        assert best is not None
        return best


class FairJointTenant:
    name = "fair_joint_tenant"

    def __init__(self, base: Controller, low_mode_id: str, queue_hard_threshold: int = 12):
        self._base = base
        self._low_mode = low_mode_id
        self._queue_hard_threshold = queue_hard_threshold

    def choose_mode(self, state: ControllerState) -> str:
        mode = self._base.choose_mode(state)
        if state.tenant_id == "premium" and mode == self._low_mode and state.queue_len < self._queue_hard_threshold:
            return state.prev_mode or mode
        return mode


def build_mode_summary(admitted_modes_df: pd.DataFrame, workload: str) -> list[ModeSummary]:
    ws = admitted_modes_df[admitted_modes_df["workload"] == workload]
    return [
        ModeSummary(
            mode_id=str(r.mode_id),
            precision=str(r.precision),
            capacity_k=int(r.capacity_k),
            latency_p95_ms=float(r.latency_p95_ms),
            energy_mean_Wh=float(r.energy_mean_Wh),
            accuracy=float(r.accuracy),
        )
        for r in ws.itertuples(index=False)
    ]