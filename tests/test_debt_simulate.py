"""借贷令牌桶试算（/debt-simulate）引擎与 HTTP 层测试：pytest 运行。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.errors import RequestValidationErrors
from app.debt import debt_simulate
from app.main import app
from app.models import DebtSimulateRequest

client = TestClient(app)


def _run(buckets, requests):
    return debt_simulate(
        DebtSimulateRequest.model_validate({"buckets": buckets, "requests": requests})
    )


def _view(result, bucket_id):
    return result.buckets[bucket_id]


# ---------- 基本借贷 ----------

def test_balance_covers_cost_deducts_normally_without_debt():
    resp = _run(
        [{"id": "A", "capacity": 10, "refill_rate": 0, "debt_limit": 5}],
        [{"id": "r1", "timestamp": 0, "cost": 4, "chain": ["A"]}],
    )
    r = resp.results[0]
    v = _view(r, "A")
    assert r.allowed is True and r.rejected_by is None
    assert (v.balance.before, v.balance.after) == (10, 6)
    assert (v.debt.before, v.debt.after) == (0, 0)


def test_shortfall_is_borrowed_up_to_debt_limit():
    # 余额 8，cost 10：借入缺口 2 -> balance 0、debt 2（上限 4 内）
    resp = _run(
        [{"id": "A", "capacity": 8, "refill_rate": 0, "debt_limit": 4}],
        [{"id": "r1", "timestamp": 0, "cost": 10, "chain": ["A"]}],
    )
    r = resp.results[0]
    v = _view(r, "A")
    assert r.allowed is True
    assert (v.balance.before, v.balance.after) == (8, 0)
    assert (v.debt.before, v.debt.after) == (0, 2)


def test_borrowing_exactly_to_the_debt_limit_is_allowed():
    # 余额 8 + 可借 4 = 12：cost 恰好 12 放行，debt 恰好到上限
    resp = _run(
        [{"id": "A", "capacity": 8, "refill_rate": 0, "debt_limit": 4}],
        [{"id": "r1", "timestamp": 0, "cost": 12, "chain": ["A"]}],
    )
    r = resp.results[0]
    assert r.allowed is True
    v = _view(r, "A")
    assert v.balance.after == 0 and v.debt.after == 4


# ---------- 补充先还债 ----------

def test_refill_repays_debt_before_adding_to_balance():
    # t=0 cost 8：balance 5->0、debt 3；t=1 以 5/s 补充 5：
    # 先还债 3，剩余 2 进入余额 -> balance 2、debt 0
    resp = _run(
        [{"id": "A", "capacity": 10, "refill_rate": 5, "initial": 5, "debt_limit": 5}],
        [
            {"id": "r1", "timestamp": 0, "cost": 8, "chain": ["A"]},
            {"id": "r2", "timestamp": 1, "cost": 1, "chain": ["A"]},
        ],
    )
    v = _view(resp.results[1], "A")
    assert (v.balance.before, v.debt.before) == (2, 0)
    assert (v.balance.after, v.debt.after) == (1, 0)


def test_partial_refill_repays_part_of_debt_and_leaves_balance_zero():
    # debt 4，补充 3：全部用于还债 -> debt 1、balance 仍为 0
    resp = _run(
        [{"id": "A", "capacity": 10, "refill_rate": 3, "initial": 0, "debt_limit": 4}],
        [
            {"id": "r1", "timestamp": 0, "cost": 4, "chain": ["A"]},
            {"id": "r2", "timestamp": 1, "cost": 1, "chain": ["A"]},
        ],
    )
    v = _view(resp.results[1], "A")
    assert (v.balance.before, v.debt.before) == (0, 1)
    # 剩余可借 3，cost 1 可借 -> 放行，debt 1->2
    assert resp.results[1].allowed is True
    assert (v.balance.after, v.debt.after) == (0, 2)


def test_refill_caps_balance_at_capacity_even_after_debt_repaid():
    # t=0 cost 10 -> debt 2（initial 8）；t=10 补充 20：还债 2 后余额封顶 10
    resp = _run(
        [{"id": "A", "capacity": 10, "refill_rate": 2, "initial": 8, "debt_limit": 5}],
        [
            {"id": "r1", "timestamp": 0, "cost": 10, "chain": ["A"]},
            {"id": "r2", "timestamp": 10, "cost": 1, "chain": ["A"]},
        ],
    )
    v = _view(resp.results[1], "A")
    assert (v.balance.before, v.debt.before) == (10, 0)


# ---------- 借贷上限拒绝 ----------

def test_request_exceeding_remaining_credit_is_rejected():
    # balance 2、debt 1、debt_limit 4：剩余可借 3，cost 6 缺口 4 > 3 -> 拒绝
    resp = _run(
        [{"id": "A", "capacity": 10, "refill_rate": 0, "initial": 2, "debt_limit": 4}],
        [
            {"id": "r0", "timestamp": 0, "cost": 3, "chain": ["A"]},  # 2 + 借入 1 -> debt 1
            {"id": "r1", "timestamp": 0, "cost": 6, "chain": ["A"]},  # 缺口 4 > 剩余可借 3
        ],
    )
    r = resp.results[1]
    assert r.allowed is False and r.rejected_by == "A"
    assert r.rejection.reason == "debt_limit_exceeded"
    assert (r.rejection.balance_before, r.rejection.debt_before) == (0, 1)
    assert r.rejection.debt_limit == 4
    assert r.rejection.available == 3
    assert r.rejection.borrowed == 6
    assert r.rejection.shortage == 3
    # 原子性：余额与债务不变
    v = _view(r, "A")
    assert (v.balance.before, v.balance.after) == (0, 0)
    assert (v.debt.before, v.debt.after) == (1, 1)


def test_zero_debt_limit_matches_plain_insufficient_balance_rejection():
    resp = _run(
        [{"id": "A", "capacity": 5, "refill_rate": 0, "initial": 2, "debt_limit": 0}],
        [{"id": "r1", "timestamp": 0, "cost": 3, "chain": ["A"]}],
    )
    r = resp.results[0]
    assert r.allowed is False and r.rejected_by == "A"
    assert r.rejection.reason == "debt_limit_exceeded"
    assert r.rejection.available == 2 and r.rejection.shortage == 1


# ---------- 永远不可达 ----------

def test_cost_above_capacity_plus_debt_limit_is_unreachable():
    # capacity 8 + debt_limit 4 = 12 < cost 13：即使满桶零债也永远不可达
    resp = _run(
        [{"id": "A", "capacity": 8, "refill_rate": 0, "debt_limit": 4}],
        [{"id": "r1", "timestamp": 0, "cost": 13, "chain": ["A"]}],
    )
    r = resp.results[0]
    assert r.allowed is False and r.rejected_by == "A"
    assert r.rejection.reason == "cost_exceeds_capacity"
    assert r.rejection.available == 12 and r.rejection.shortage == 1


def test_unreachable_request_stays_rejected_after_refill():
    # 零速率不可达：无论等多久补充都不会发生
    resp = _run(
        [{"id": "A", "capacity": 5, "refill_rate": 0, "debt_limit": 2}],
        [{"id": "r1", "timestamp": 100, "cost": 8, "chain": ["A"]}],
    )
    assert resp.results[0].rejection.reason == "cost_exceeds_capacity"


# ---------- 链上原子性 ----------

def test_chain_atomic_rejection_leaves_balance_and_debt_untouched():
    # A 可借后满足（10 + 借 2），B 容量+债务上限只有 11 < cost 12 永远不可达
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0, "debt_limit": 5},
            {"id": "B", "capacity": 8, "refill_rate": 0, "debt_limit": 3},
        ],
        [{"id": "r1", "timestamp": 0, "cost": 12, "chain": ["A", "B"]}],
    )
    r = resp.results[0]
    assert r.allowed is False and r.rejected_by == "B"
    assert r.rejection.reason == "cost_exceeds_capacity"
    va, vb = _view(r, "A"), _view(r, "B")
    assert (va.balance.before, va.balance.after, va.debt.after) == (10, 10, 0)
    assert (vb.balance.before, vb.balance.after, vb.debt.after) == (8, 8, 0)


def test_rejected_by_first_failing_bucket_in_chain_order():
    # A：cost 10 ≤ capacity+limit=12（结构可达），但余额 0、已欠 1，
    # 再借 10 会超过上限 2 -> debt_limit_exceeded；B 结构不可达。
    # 按链顺序先撞 A，不再检查 B。
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0, "initial": 1, "debt_limit": 2},
            {"id": "B", "capacity": 5, "refill_rate": 0, "debt_limit": 1},
        ],
        [
            {"id": "r0", "timestamp": 0, "cost": 2, "chain": ["A"]},  # 1 + 借 1 -> debt 1
            {"id": "r1", "timestamp": 0, "cost": 10, "chain": ["A", "B"]},
        ],
    )
    r = resp.results[1]
    assert r.rejected_by == "A" and r.rejection.reason == "debt_limit_exceeded"


def test_chain_allow_borrows_from_every_bucket():
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0, "debt_limit": 5},
            {"id": "B", "capacity": 8, "refill_rate": 0, "debt_limit": 4},
        ],
        [{"id": "r1", "timestamp": 0, "cost": 10, "chain": ["A", "B"]}],
    )
    r = resp.results[0]
    assert r.allowed is True
    assert (_view(r, "A").balance.after, _view(r, "A").debt.after) == (0, 0)
    assert (_view(r, "B").balance.after, _view(r, "B").debt.after) == (0, 2)


# ---------- 顺序与终态 ----------

def test_out_of_order_timestamps_simulated_chronologically():
    # 乱序输入：r_late(t=2) 先出现，r_early(t=0) 后出现
    # t=0：balance 0 + 借 3（上限 5）-> debt 3
    # t=2：以 2/s 补充 4，先还债 3、余 1 进余额；cost 1 正常扣 -> balance 0、debt 0
    resp = _run(
        [{"id": "A", "capacity": 10, "refill_rate": 2, "initial": 0, "debt_limit": 5}],
        [
            {"id": "r_late", "timestamp": 2, "cost": 1, "chain": ["A"]},
            {"id": "r_early", "timestamp": 0, "cost": 3, "chain": ["A"]},
        ],
    )
    assert resp.results[0].request_id == "r_late"
    v = _view(resp.results[0], "A")
    assert resp.results[0].allowed is True
    assert (v.balance.before, v.debt.before, v.balance.after, v.debt.after) == (1, 0, 0, 0)
    assert _view(resp.results[1], "A").debt.after == 3
    assert resp.allowed_count == 2


def test_final_snapshot_refills_all_buckets_to_last_timestamp():
    # r1 t=0：A 借 2（balance 8->0, debt 2）；r2 t=2 只引用 A：
    # A 补 2 先还清债务 2，没有剩余进余额，再借 1 -> 终态 balance 0、debt 1。
    # B 从未被引用，终态时推进到 t=2（初始满桶，补充封顶，仍 8）。
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 1, "initial": 8, "debt_limit": 5},
            {"id": "B", "capacity": 8, "refill_rate": 2, "debt_limit": 3},
        ],
        [
            {"id": "r1", "timestamp": 0, "cost": 10, "chain": ["A"]},
            {"id": "r2", "timestamp": 2, "cost": 1, "chain": ["A"]},
        ],
    )
    final = {b.id: b for b in resp.buckets}
    # t=0：A 8->0、debt 2；r2 t=2：补 2 先还债 2 -> balance 0、debt 0，再扣 1
    assert (final["A"].balance, final["A"].debt, final["A"].net) == (0, 1, -1)
    assert final["A"].debt_headroom == 4 and final["A"].available == 4
    assert (final["B"].balance, final["B"].debt) == (8, 0)
    assert final["B"].debt_headroom == 3 and final["B"].available == 11


def test_results_returned_in_original_order_with_index():
    resp = _run(
        [{"id": "A", "capacity": 100, "refill_rate": 0, "debt_limit": 0}],
        [
            {"id": "r3", "timestamp": 3, "cost": 1, "chain": ["A"]},
            {"id": "r1", "timestamp": 1, "cost": 1, "chain": ["A"]},
            {"id": "r2", "timestamp": 2, "cost": 1, "chain": ["A"]},
        ],
    )
    assert [r.request_id for r in resp.results] == ["r3", "r1", "r2"]
    assert [r.index for r in resp.results] == [0, 1, 2]


# ---------- 输入校验 ----------

def _validate_error(payload):
    with pytest.raises(RequestValidationErrors) as exc:
        DebtSimulateRequest.model_validate(payload)
    return exc.value.errors


def test_negative_debt_limit_rejected_by_pydantic():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DebtSimulateRequest.model_validate(
            {
                "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0, "debt_limit": -1}],
                "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["A"]}],
            }
        )


def test_missing_debt_limit_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DebtSimulateRequest.model_validate(
            {
                "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
                "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["A"]}],
            }
        )


def test_extra_bucket_field_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DebtSimulateRequest.model_validate(
            {
                "buckets": [
                    {"id": "A", "capacity": 1, "refill_rate": 0, "debt_limit": 1, "x": 1}
                ],
                "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["A"]}],
            }
        )


def test_duplicate_bucket_ids_rejected():
    errors = _validate_error(
        {
            "buckets": [
                {"id": "A", "capacity": 1, "refill_rate": 0, "debt_limit": 1},
                {"id": "A", "capacity": 2, "refill_rate": 0, "debt_limit": 1},
            ],
            "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["A"]}],
        }
    )
    assert all(e["type"] == "duplicate_bucket" for e in errors)


def test_unknown_bucket_reference_rejected():
    errors = _validate_error(
        {
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0, "debt_limit": 1}],
            "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["A", "X"]}],
        }
    )
    assert any(e["type"] == "unknown_bucket_ref" and e["input"] == "X" for e in errors)


def test_duplicate_request_ids_rejected():
    errors = _validate_error(
        {
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0, "debt_limit": 1}],
            "requests": [
                {"id": "r", "timestamp": 0, "cost": 1, "chain": ["A"]},
                {"id": "r", "timestamp": 1, "cost": 1, "chain": ["A"]},
            ],
        }
    )
    assert any(e["type"] == "duplicate_request" for e in errors)


def test_non_positive_cost_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DebtSimulateRequest.model_validate(
            {
                "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0, "debt_limit": 1}],
                "requests": [{"id": "r", "timestamp": 0, "cost": 0, "chain": ["A"]}],
            }
        )


# ---------- HTTP ----------

def test_http_debt_simulate_ok():
    response = client.post(
        "/debt-simulate",
        json={
            "buckets": [
                {"id": "api", "capacity": 10, "refill_rate": 1, "debt_limit": 5},
                {"id": "db", "capacity": 8, "refill_rate": 2, "debt_limit": 4},
            ],
            "requests": [
                {"id": "r1", "timestamp": 0, "cost": 10, "chain": ["api", "db"]},
                {"id": "r2", "timestamp": 4, "cost": 13, "chain": ["api", "db"]},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    r1, r2 = data["results"]
    # r1：api 10->0 无债；db 8->0、借入 2
    assert r1["allowed"] is True
    assert r1["buckets"]["db"]["debt"]["after"] == 2.0
    # r2：t=4。api 补 4 后余额 4，借入 9 超过上限 5 -> 拒绝；
    # db 补 8：先还债 2、余 6 进余额；原子拒绝后 db 保持 6
    assert r2["allowed"] is False
    assert r2["rejected_by"] == "api"
    assert r2["rejection"]["reason"] == "debt_limit_exceeded"
    assert r2["buckets"]["db"]["balance"]["before"] == 6.0
    assert r2["buckets"]["db"]["balance"]["after"] == 6.0
    assert {b["id"] for b in data["buckets"]} == {"api", "db"}


def test_http_validation_error_is_422():
    response = client.post(
        "/debt-simulate",
        json={
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0, "debt_limit": -1}],
            "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["A"]}],
        },
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_http_unknown_bucket_ref_is_422():
    response = client.post(
        "/debt-simulate",
        json={
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0, "debt_limit": 1}],
            "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["GHOST"]}],
        },
    )
    assert response.status_code == 422
    assert any(e["type"] == "unknown_bucket_ref" for e in response.json()["detail"])


def test_http_root_lists_debt_simulate():
    root = client.get("/").json()
    assert root["debt_simulate"] == "/debt-simulate"
