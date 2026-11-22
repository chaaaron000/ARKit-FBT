"""떨림을 줄이는 One Euro Filter (https://gery.casiez.net/1euro/)."""

from __future__ import annotations

import math

import numpy as np

from .mathutil import quat_normalize


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """min_cutoff: 낮을수록 가만히 있을 때 떨림이 줄지만 느려진다.
    beta: 높을수록 빠르게 움직일 때 지연이 줄지만 떨림이 늘어난다."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None
        self._v: np.ndarray | None = None

    @property
    def velocity(self) -> np.ndarray | None:
        """필터 출력의 변화 속도 (단위/초). 예측에 쓴다."""
        return None if self._v is None else self._v.copy()

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            self._v = np.zeros_like(x)
            return x.copy()
        dt = t - self._t
        if dt <= 0:
            # 같은 시각(또는 순서 뒤바뀜): 마지막 값을 유지
            return self._x.copy()
        dt = min(dt, 0.5)
        dx = (x - self._x) / dt
        a_d = _alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = _alpha(dt, cutoff)
        x_hat = a * x + (1 - a) * self._x
        v = (x_hat - self._x) / dt
        a_v = _alpha(dt, self.d_cutoff * 4)
        self._v = a_v * v + (1 - a_v) * (self._v if self._v is not None else np.zeros_like(x))
        self._x, self._dx, self._t = x_hat, dx_hat, t
        return x_hat.copy()


class QuatOneEuroFilter:
    """쿼터니언 성분에 One Euro를 적용하고 다시 정규화한다 (작은 회전 변화에 충분히 정확)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3, d_cutoff: float = 1.0):
        self._f = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._last: np.ndarray | None = None

    @property
    def min_cutoff(self) -> float:
        return self._f.min_cutoff

    @min_cutoff.setter
    def min_cutoff(self, v: float) -> None:
        self._f.min_cutoff = v

    @property
    def beta(self) -> float:
        return self._f.beta

    @beta.setter
    def beta(self, v: float) -> None:
        self._f.beta = v

    def reset(self) -> None:
        self._f.reset()
        self._last = None

    def __call__(self, q, t: float) -> np.ndarray:
        q = quat_normalize(q)
        if self._last is not None and float(np.dot(q, self._last)) < 0:
            q = -q  # 같은 반구로 맞춰야 평균이 의미 있다
        out = quat_normalize(self._f(q, t))
        self._last = out
        return out
