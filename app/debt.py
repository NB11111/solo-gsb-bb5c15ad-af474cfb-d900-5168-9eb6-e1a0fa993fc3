"""借贷令牌桶试算：连续补充先还债、余额不足可在 debt_limit 内借入的链上原子扣减。

每个桶除 capacity / refill_rate 外还有累计债务 debt（初始为 0）与债务上限
debt_limit。状态满足不变量：balance 与 debt 至少一个为 0，且
0 ≤ balance ≤ capacity、0 ≤ debt ≤ debt_limit。

- **连续补充先还债**：每次判定先按时间差连续补充；若桶有债务，补充的令牌
  先用于偿还债务（debt 下降），还清之后的剩余才进入可用余额（balance，封顶
  capacity）。
- **借贷扣减**：余额足以支付 cost 时正常扣减；否则借入缺口——扣减后
  balance = 0、debt += cost - balance。累计债务（含本次借入）不得超过
  debt_limit。
- **不可达**：cost > capacity + debt_limit 的请求对该桶永远不可达，因为
  可用余额与借贷额度的合计上限都不足以支付。
- **链上原子**：链上任一桶无法在借贷上限内满足成本，整笔请求拒绝，链上
  所有桶的余额与债务保持不变（补充还债作为时间推进照常发生）。
"""
from __future__ import annotations

from app.models import (
    EPS,
    BucketBalance,
    DebtBalance,
    DebtBucketFinalState,
    DebtRejection,
    DebtSimResult,
    DebtSimulateRequest,
    DebtSimulateResponse,
)
from app.simulation import _round


class _BucketState:
    """可借贷桶的运行时状态。"""

    __slots__ = ("balance", "debt", "last_time")

    def __init__(self, balance: float, last_time: float):
        self.balance = balance
        self.debt = 0.0
        self.last_time = last_time


def debt_simulate(payload: DebtSimulateRequest) -> DebtSimulateResponse:
    requests = payload.requests

    t_start = min(request.timestamp for request in requests)
    t_end = max(request.timestamp for request in requests)

    capacity = {b.id: float(b.capacity) for b in payload.buckets}
    refill_rate = {b.id: float(b.refill_rate) for b in payload.buckets}
    debt_limit = {b.id: float(b.debt_limit) for b in payload.buckets}
    bucket_ids = [b.id for b in payload.buckets]
    states = {
        b.id: _BucketState(float(b.capacity if b.initial is None else b.initial), t_start)
        for b in payload.buckets
    }

    def advance(bucket_id: str, now: float) -> None:
        """把某桶连续补充到时刻 now：补充量先偿还债务，剩余进入可用余额。"""
        state = states[bucket_id]
        elapsed = now - state.last_time
        if elapsed > 0:
            added = refill_rate[bucket_id] * elapsed
            if state.debt > 0.0:
                repaid = min(state.debt, added)
                state.debt -= repaid
                added -= repaid
            if added > 0.0:
                state.balance = min(capacity[bucket_id], state.balance + added)
        state.last_time = now

    # 内部按时间顺序处理；同一时刻按原始输入顺序处理
    order = sorted(range(len(requests)), key=lambda i: (requests[i].timestamp, i))

    results_by_index: dict[int, DebtSimResult] = {}

    for i in order:
        request = requests[i]
        now = float(request.timestamp)
        cost = float(request.cost)

        # 先对链上所有桶做连续补充（还债优先；无论最终是否放行，时间都会推进）
        for bucket_id in request.chain:
            advance(bucket_id, now)

        before: dict[str, tuple[float, float]] = {
            bucket_id: (states[bucket_id].balance, states[bucket_id].debt)
            for bucket_id in request.chain
        }

        # 链上第一个无法在借贷上限内满足成本的桶即拒绝桶：
        # 1) cost 超过 capacity + debt_limit：结构上永远不可达；
        # 2) 否则缺口（cost - 补充后余额）不得把累计债务推过 debt_limit。
        rejected_by: str | None = None
        reason: str | None = None
        for bucket_id in request.chain:
            balance_before, debt_before = before[bucket_id]
            if cost > capacity[bucket_id] + debt_limit[bucket_id] + EPS:
                rejected_by, reason = bucket_id, "cost_exceeds_capacity"
                break
            if debt_before + max(0.0, cost - balance_before) > debt_limit[bucket_id] + EPS:
                rejected_by, reason = bucket_id, "debt_limit_exceeded"
                break

        allowed = rejected_by is None

        if allowed:
            # 原子扣减：余额不足的部分借入，累计债务不得超过 debt_limit
            for bucket_id in request.chain:
                state = states[bucket_id]
                if state.balance + EPS >= cost:
                    state.balance -= cost
                else:
                    state.debt += cost - state.balance
                    state.balance = 0.0

        bucket_views: dict[str, DebtBalance] = {}
        for bucket_id in request.chain:
            state = states[bucket_id]
            balance_before, debt_before = before[bucket_id]
            if allowed:
                balance_after, debt_after = state.balance, state.debt
            else:
                # 原子性：拒绝时链上所有桶的余额与债务保持不变
                balance_after, debt_after = balance_before, debt_before
            bucket_views[bucket_id] = DebtBalance(
                balance=BucketBalance(
                    before=_round(balance_before), after=_round(balance_after)
                ),
                debt=BucketBalance(
                    before=_round(debt_before), after=_round(debt_after)
                ),
            )

        rejection: DebtRejection | None = None
        if rejected_by is not None:
            balance_before, debt_before = before[rejected_by]
            borrowed = max(0.0, cost - balance_before)
            available = balance_before + (debt_limit[rejected_by] - debt_before)
            rejection = DebtRejection(
                bucket_id=rejected_by,
                reason=reason,
                cost=_round(cost),
                balance_before=_round(balance_before),
                debt_before=_round(debt_before),
                debt_limit=_round(debt_limit[rejected_by]),
                available=_round(available),
                borrowed=_round(borrowed),
                shortage=_round(max(0.0, cost - available)),
            )

        results_by_index[i] = DebtSimResult(
            request_id=request.id,
            index=i,
            timestamp=_round(now),
            cost=_round(cost),
            allowed=allowed,
            buckets=bucket_views,
            rejected_by=rejected_by,
            rejection=rejection,
        )

    results = [results_by_index[i] for i in range(len(requests))]
    allowed_count = sum(1 for r in results if r.allowed)

    # 终态快照：把所有桶补充（还债）到最后请求时刻
    final_states: list[DebtBucketFinalState] = []
    for bucket_id in bucket_ids:
        advance(bucket_id, t_end)
        state = states[bucket_id]
        headroom = debt_limit[bucket_id] - state.debt
        final_states.append(
            DebtBucketFinalState(
                id=bucket_id,
                capacity=_round(capacity[bucket_id]),
                refill_rate=_round(refill_rate[bucket_id]),
                debt_limit=_round(debt_limit[bucket_id]),
                balance=_round(state.balance),
                debt=_round(state.debt),
                net=_round(state.balance - state.debt),
                debt_headroom=_round(headroom),
                available=_round(state.balance + headroom),
            )
        )

    return DebtSimulateResponse(
        allowed_count=allowed_count,
        rejected_count=len(results) - allowed_count,
        results=results,
        buckets=final_states,
    )
