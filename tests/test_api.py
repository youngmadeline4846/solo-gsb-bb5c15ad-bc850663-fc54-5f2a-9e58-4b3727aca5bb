"""End-to-end API tests."""
import pytest

from tests.conftest import make_request


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_solve_optimal(client):
    r = client.post("/api/v1/blends/solve", json=make_request())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    assert body["plan_id"] is None
    assert body["total_cost"] == pytest.approx(1500.0, abs=1e-6)
    blend = {a["source_id"]: a["volume"] for a in body["blend"]}
    assert blend["W1"] == pytest.approx(500.0, abs=1e-6)
    assert blend["W2"] == pytest.approx(500.0, abs=1e-6)
    assert body["indicators"]["calcium"] == pytest.approx(60.0, abs=1e-6)
    assert body["indicators"]["magnesium"] == pytest.approx(17.5, abs=1e-6)
    assert body["indicators"]["alkalinity"] == pytest.approx(165.0, abs=1e-6)


def test_solve_tie_break_lexicographic(client):
    payload = {
        "target_volume": 100,
        "sources": [
            {"id": "A", "max_volume": 100, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
            {"id": "B", "max_volume": 100, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
            {"id": "C", "max_volume": 60, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
        ],
    }
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    blend = {a["source_id"]: a["volume"] for a in r.json()["blend"]}
    assert blend["A"] == pytest.approx(0.0, abs=1e-6)
    assert blend["B"] == pytest.approx(40.0, abs=1e-6)
    assert blend["C"] == pytest.approx(60.0, abs=1e-6)


def test_solve_infeasible_indicator(client):
    r = client.post(
        "/api/v1/blends/solve", json=make_request(calcium={"min": 70, "max": 90})
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    ca = body["achievable_ranges"]["calcium"]
    assert ca["max"] == pytest.approx(68.0, abs=1e-6)
    assert ca["requested_min"] == 70
    assert "calcium" in body["message"]


def test_solve_infeasible_volume(client):
    r = client.post("/api/v1/blends/solve", json=make_request(target_volume=2000))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    assert body["achievable_ranges"]["volume"]["max"] == pytest.approx(1400.0)


def test_create_and_get_plan(client):
    r = client.post("/api/v1/blends", json=make_request(idempotency_key="plan-1"))
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "optimal"
    assert body["plan_id"] is not None
    assert body["created_at"]

    g = client.get(f"/api/v1/blends/{body['plan_id']}")
    assert g.status_code == 200
    assert g.json() == body


def test_get_plan_not_found(client):
    assert client.get("/api/v1/blends/9999").status_code == 404


def test_idempotent_replay(client):
    payload = make_request(idempotency_key="replay-me")
    r1 = client.post("/api/v1/blends", json=payload)
    r2 = client.post("/api/v1/blends", json=payload)
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r2.headers.get("x-idempotent-replay") == "true"
    assert r1.json() == r2.json()


def test_idempotency_conflict(client):
    client.post("/api/v1/blends", json=make_request(idempotency_key="dup"))
    r = client.post(
        "/api/v1/blends",
        json=make_request(idempotency_key="dup", target_volume=500),
    )
    assert r.status_code == 409


def test_idempotency_key_via_header(client):
    r = client.post(
        "/api/v1/blends",
        json=make_request(),
        headers={"Idempotency-Key": "header-key"},
    )
    assert r.status_code == 201
    again = client.post(
        "/api/v1/blends",
        json=make_request(),
        headers={"Idempotency-Key": "header-key"},
    )
    assert again.status_code == 200
    assert again.json()["plan_id"] == r.json()["plan_id"]


def test_missing_idempotency_key(client):
    r = client.post("/api/v1/blends", json=make_request())
    assert r.status_code == 422


def test_infeasible_plan_is_persisted(client):
    r = client.post(
        "/api/v1/blends",
        json=make_request(idempotency_key="infeasible", calcium={"min": 70, "max": 90}),
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "infeasible"
    g = client.get(f"/api/v1/blends/{body['plan_id']}")
    assert g.status_code == 200
    assert g.json()["status"] == "infeasible"
    assert "calcium" in g.json()["achievable_ranges"]


def test_validation_source_count(client):
    one = make_request()
    one["sources"] = one["sources"][:1]
    assert client.post("/api/v1/blends/solve", json=one).status_code == 422

    seven = make_request()
    seven["sources"] = [
        {"id": f"S{i}", "max_volume": 10, "cost": 1, "calcium": 1,
         "magnesium": 1, "alkalinity": 1}
        for i in range(7)
    ]
    assert client.post("/api/v1/blends/solve", json=seven).status_code == 422


def test_validation_bad_range_and_duplicate_ids(client):
    bad_range = make_request(calcium={"min": 60, "max": 40})
    assert client.post("/api/v1/blends/solve", json=bad_range).status_code == 422

    dup = make_request()
    dup["sources"][1]["id"] = "W1"
    assert client.post("/api/v1/blends/solve", json=dup).status_code == 422

    neg = make_request(target_volume=-5)
    assert client.post("/api/v1/blends/solve", json=neg).status_code == 422


# --------------------------------------------------------------------------- #
# supply-reduction contingencies
# --------------------------------------------------------------------------- #

def _save_optimal_plan(client, key="cont-1", **overrides):
    r = client.post("/api/v1/blends", json=make_request(idempotency_key=key, **overrides))
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "optimal"
    return body


def test_contingency_feasible(client):
    plan = _save_optimal_plan(client)
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"W1": 400}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    assert body["plan_id"] == plan["plan_id"]
    assert body["total_volume"] == 1000
    assert body["total_cost"] == pytest.approx(1600.0, abs=1e-6)
    assert body["cost_delta"] == pytest.approx(100.0, abs=1e-6)
    assert body["total_adjustment"] == pytest.approx(200.0, abs=1e-6)
    rows = {a["source_id"]: a for a in body["blend"]}
    assert rows["W1"]["original_volume"] == pytest.approx(500.0, abs=1e-6)
    assert rows["W1"]["new_volume"] == pytest.approx(400.0, abs=1e-6)
    assert rows["W1"]["volume_delta"] == pytest.approx(-100.0, abs=1e-6)
    assert rows["W1"]["new_cap"] == pytest.approx(400.0, abs=1e-6)
    assert rows["W1"]["cost"] == pytest.approx(400.0, abs=1e-6)
    assert rows["W2"]["original_volume"] == pytest.approx(500.0, abs=1e-6)
    assert rows["W2"]["new_volume"] == pytest.approx(600.0, abs=1e-6)
    assert rows["W2"]["volume_delta"] == pytest.approx(100.0, abs=1e-6)
    # W2 was not listed: its new cap is the original max_volume
    assert rows["W2"]["new_cap"] == pytest.approx(800.0, abs=1e-6)
    assert body["indicators"]["calcium"] == pytest.approx(52.0, abs=1e-6)
    assert body["indicators"]["magnesium"] == pytest.approx(15.0, abs=1e-6)
    assert body["indicators"]["alkalinity"] == pytest.approx(148.0, abs=1e-6)


def test_contingency_non_binding_cap_keeps_plan(client):
    plan = _save_optimal_plan(client, key="cont-keep")
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"W2": 800}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total_adjustment"] == pytest.approx(0.0, abs=1e-9)
    assert body["cost_delta"] == pytest.approx(0.0, abs=1e-9)
    assert body["total_cost"] == pytest.approx(1500.0, abs=1e-6)


def test_contingency_does_not_modify_or_persist(client):
    plan = _save_optimal_plan(client, key="cont-ro")
    client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"W1": 400}
    )
    # the stored plan is untouched and fetching it again returns the same body
    g = client.get(f"/api/v1/blends/{plan['plan_id']}")
    assert g.status_code == 200
    assert g.json() == plan


def test_contingency_infeasible_reports_reason(client):
    plan = _save_optimal_plan(client, key="cont-inf")
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency",
        json={"W1": 100, "W2": 800},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    assert body["plan_id"] == plan["plan_id"]
    assert body["achievable_ranges"]["volume"]["max"] == pytest.approx(900.0)
    assert "below the target" in body["message"]
    # no plan was saved or modified on infeasibility either
    assert client.get(f"/api/v1/blends/{plan['plan_id']}").json() == plan


def test_contingency_unknown_plan_404(client):
    r = client.post("/api/v1/blends/9999/contingency", json={"W1": 100})
    assert r.status_code == 404


def test_contingency_non_optimal_plan_422(client):
    r = client.post(
        "/api/v1/blends",
        json=make_request(idempotency_key="cont-bad-status", calcium={"min": 70, "max": 90}),
    )
    assert r.status_code == 201
    plan_id = r.json()["plan_id"]
    resp = client.post(f"/api/v1/blends/{plan_id}/contingency", json={"W1": 100})
    assert resp.status_code == 422
    assert "optimal" in resp.json()["detail"]


def test_contingency_validation_errors(client):
    plan = _save_optimal_plan(client, key="cont-validate")
    url = f"/api/v1/blends/{plan['plan_id']}/contingency"
    # empty map
    assert client.post(url, json={}).status_code == 422
    # unknown source id
    assert client.post(url, json={"NOPE": 100}).status_code == 422
    # cap above the original max
    assert client.post(url, json={"W1": 601}).status_code == 422
    # negative cap
    assert client.post(url, json={"W1": -0.001}).status_code == 422
    # non-numeric / null values (numeric strings coerce, like other fields)
    assert client.post(url, json={"W1": "lots"}).status_code == 422
    assert client.post(url, json={"W1": None}).status_code == 422
    # body must be an object
    assert client.post(url, json=[["W1", 100]]).status_code == 422
    # zero cap is legal
    assert client.post(url, json={"W1": 0}).status_code == 200
    # a cap equal to the original max is legal
    assert client.post(url, json={"W1": 600}).status_code == 200


def test_contingency_lexicographic_tie_break(client):
    payload = {
        "target_volume": 100,
        "sources": [
            {"id": "A", "max_volume": 100, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
            {"id": "B", "max_volume": 100, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
            {"id": "C", "max_volume": 60, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
        ],
        "idempotency_key": "cont-lex",
    }
    plan = client.post("/api/v1/blends", json=payload).json()
    # saved blend (lexicographic cost minimum): A=0, B=40, C=60
    # cap C at 30: 30 units must leave C (minimum change 60); the freed
    # volume can go to A or B, and the id-ascending lex-min puts it in B.
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency",
        json={"C": 30},
    )
    assert r.status_code == 200
    blend = {a["source_id"]: a["new_volume"] for a in r.json()["blend"]}
    assert blend["A"] == pytest.approx(0.0, abs=1e-6)
    assert blend["B"] == pytest.approx(70.0, abs=1e-6)
    assert blend["C"] == pytest.approx(30.0, abs=1e-6)
    assert r.json()["total_adjustment"] == pytest.approx(60.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# batch_size end-to-end
# --------------------------------------------------------------------------- #

def _batch_request(idempotency_key=None):
    payload = {
        "target_volume": 1000,
        "calcium": {"min": 40, "max": 80},
        "sources": [
            # continuous high-calcium, expensive
            {"id": "A", "max_volume": 1000, "cost": 2.0, "calcium": 100,
             "magnesium": 10, "alkalinity": 100},
            # batched low-calcium, cheap
            {"id": "B", "max_volume": 1000, "cost": 1.0, "calcium": 20,
             "magnesium": 10, "alkalinity": 100, "batch_size": 300},
        ],
    }
    if idempotency_key is not None:
        payload["idempotency_key"] = idempotency_key
    return payload


def test_solve_mixed_integer_batch(client):
    r = client.post("/api/v1/blends/solve", json=_batch_request())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    blend = {a["source_id"]: a["volume"] for a in body["blend"]}
    # continuous optimum B=500 is not 300-aligned; MIP must pick B=600
    assert blend["A"] == pytest.approx(400.0, abs=1e-6)
    assert blend["B"] == pytest.approx(600.0, abs=1e-6)
    assert body["total_cost"] == pytest.approx(1400.0, abs=1e-6)
    assert body["indicators"]["calcium"] == pytest.approx(52.0, abs=1e-6)


def test_solve_non_divisible_batch_cap(client):
    payload = _batch_request()
    payload["sources"][1]["max_volume"] = 850  # floor(850/300) = 2 batches
    payload.pop("calcium")
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    blend = {a["source_id"]: a["volume"] for a in r.json()["blend"]}
    assert blend["B"] == pytest.approx(600.0, abs=1e-6)
    assert blend["A"] == pytest.approx(400.0, abs=1e-6)


def test_solve_integer_lexicographic_tie(client):
    payload = {
        "target_volume": 500,
        "sources": [
            {"id": "C", "max_volume": 300, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 100},
            {"id": "A", "max_volume": 300, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 100},
            {"id": "B", "max_volume": 300, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 100},
        ],
    }
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    blend = {a["source_id"]: a["volume"] for a in r.json()["blend"]}
    assert blend["A"] == pytest.approx(0.0, abs=1e-6)
    assert blend["B"] == pytest.approx(200.0, abs=1e-6)
    assert blend["C"] == pytest.approx(300.0, abs=1e-6)


def test_solve_discrete_infeasible_uses_mip_ranges(client):
    # Both sources batched at 300; 1000 cannot be assembled (largest is 900).
    payload = {
        "target_volume": 1000,
        "sources": [
            {"id": "W1", "max_volume": 600, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 300},
            {"id": "W2", "max_volume": 600, "cost": 2, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 300},
        ],
    }
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    assert body["achievable_ranges"]["volume"]["max"] == pytest.approx(900.0)
    assert "batch-size" in body["message"]
    assert body["achievable_ranges"]["calcium"]["min"] >= 0


@pytest.mark.parametrize("bad", [0, -1, -0.5, "abc", True, False])
def test_batch_size_validation(client, bad):
    payload = make_request()
    payload["sources"][0]["batch_size"] = bad
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_batch_size_must_be_finite(client, token):
    # NaN/+inf/-inf parse as floats but must be rejected; send them as raw
    # JSON tokens because the JSON encoder refuses to serialize floats NaN.
    raw = (
        '{"target_volume": 1000, "sources": ['
        '{"id": "W1", "max_volume": 600, "cost": 1.0, "calcium": 100,'
        ' "magnesium": 30, "alkalinity": 250, "batch_size": ' + token + '},'
        '{"id": "W2", "max_volume": 800, "cost": 2.0, "calcium": 20,'
        ' "magnesium": 5, "alkalinity": 80}'
        ']}'
    )
    r = client.post(
        "/api/v1/blends/solve", content=raw,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422, r.text


def test_batch_size_null_means_unset(client):
    payload = make_request()
    payload["sources"][0]["batch_size"] = None  # explicit null == continuous
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200, r.text


def test_batch_size_accepts_finite_positive_values(client):
    payload = make_request()
    payload["sources"][0]["batch_size"] = "250"  # numeric strings coerce
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200, r.text


def test_batch_plan_persisted_and_replayed(client):
    payload = _batch_request(idempotency_key="batch-plan-1")
    r1 = client.post("/api/v1/blends", json=payload)
    assert r1.status_code == 201
    assert r1.json()["status"] == "optimal"
    r2 = client.post("/api/v1/blends", json=payload)
    assert r2.status_code == 200
    assert r2.headers.get("x-idempotent-replay") == "true"
    assert r2.json() == r1.json()
    g = client.get(f"/api/v1/blends/{r1.json()['plan_id']}")
    assert g.status_code == 200
    assert g.json() == r1.json()


def test_batch_size_part_of_idempotency_fingerprint(client):
    with_batch = _batch_request(idempotency_key="fp-key")
    r1 = client.post("/api/v1/blends", json=with_batch)
    assert r1.status_code == 201
    # same key, batch_size dropped: different request -> conflict
    without_batch = _batch_request(idempotency_key="fp-key")
    without_batch["sources"][1].pop("batch_size")
    r2 = client.post("/api/v1/blends", json=without_batch)
    assert r2.status_code == 409


def test_infeasible_batch_plan_is_persisted(client):
    payload = {
        "target_volume": 1000,
        "idempotency_key": "batch-infeasible",
        "sources": [
            {"id": "W1", "max_volume": 600, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 300},
            {"id": "W2", "max_volume": 600, "cost": 2, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 300},
        ],
    }
    r = client.post("/api/v1/blends", json=payload)
    assert r.status_code == 201
    assert r.json()["status"] == "infeasible"
    g = client.get(f"/api/v1/blends/{r.json()['plan_id']}")
    assert g.status_code == 200
    assert g.json()["achievable_ranges"]["volume"]["max"] == pytest.approx(900.0)


def test_contingency_preserves_batch_size(client):
    plan = client.post(
        "/api/v1/blends", json=_batch_request(idempotency_key="batch-cont-1")
    ).json()
    assert plan["status"] == "optimal"
    # cut B to 350: at most one 300-batch; A takes the other 300
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"B": 350}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    rows = {a["source_id"]: a for a in body["blend"]}
    assert rows["A"]["new_volume"] == pytest.approx(700.0, abs=1e-6)
    assert rows["B"]["new_volume"] == pytest.approx(300.0, abs=1e-6)
    assert rows["B"]["new_cap"] == pytest.approx(350.0, abs=1e-6)
    assert body["total_adjustment"] == pytest.approx(600.0, abs=1e-6)
    assert body["indicators"]["calcium"] == pytest.approx(76.0, abs=1e-6)


def test_contingency_batch_infeasible(client):
    # V=900 = W1 300 (ca 100) + W2 600 (ca 20), calcium [40,60]; removing W1
    # makes calcium 20 no matter how the batches are arranged.
    payload = {
        "target_volume": 900,
        "calcium": {"min": 40, "max": 60},
        "idempotency_key": "batch-cont-inf",
        "sources": [
            {"id": "W1", "max_volume": 600, "cost": 1, "calcium": 100,
             "magnesium": 10, "alkalinity": 100, "batch_size": 300},
            {"id": "W2", "max_volume": 900, "cost": 2, "calcium": 20,
             "magnesium": 10, "alkalinity": 100, "batch_size": 300},
        ],
    }
    plan = client.post("/api/v1/blends", json=payload).json()
    assert plan["status"] == "optimal"
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"W1": 0}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    assert body["achievable_ranges"]["calcium"]["max"] == pytest.approx(20.0, abs=1e-6)

# --------------------------------------------------------------------------- #
# minimum_run_volume / minimum_active_sources end-to-end
# --------------------------------------------------------------------------- #

def _commission_request(idempotency_key=None, minimum_active_sources=None):
    payload = {
        "target_volume": 500,
        "calcium": {"min": 15, "max": 67},
        "sources": [
            # cheap high-calcium water with a 300 m3 commissioning threshold
            {"id": "A", "max_volume": 500, "cost": 1.0, "calcium": 100,
             "magnesium": 10, "alkalinity": 100,
             "minimum_run_volume": 300},
            {"id": "B", "max_volume": 500, "cost": 2.0, "calcium": 20,
             "magnesium": 10, "alkalinity": 100},
        ],
    }
    if minimum_active_sources is not None:
        payload["minimum_active_sources"] = minimum_active_sources
    if idempotency_key is not None:
        payload["idempotency_key"] = idempotency_key
    return payload


def test_solve_minimum_run_zero_or_threshold(client):
    r = client.post("/api/v1/blends/solve", json=_commission_request())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    blend = {a["source_id"]: a["volume"] for a in body["blend"]}
    # the continuous optimum A=250 lies in the forbidden band (0, 300);
    # the MIP shuts A off and takes all 500 from B
    assert blend["A"] == pytest.approx(0.0, abs=1e-6)
    assert blend["B"] == pytest.approx(500.0, abs=1e-6)
    assert body["total_cost"] == pytest.approx(1000.0, abs=1e-6)


def test_solve_minimum_active_sources(client):
    # Two threshold sources of 300, V=700, K=2 forces both to run; the
    # cost optimum is A=400, B=300.
    payload = {
        "target_volume": 700,
        "minimum_active_sources": 2,
        "sources": [
            {"id": "A", "max_volume": 700, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100,
             "minimum_run_volume": 300},
            {"id": "B", "max_volume": 700, "cost": 2, "calcium": 50,
             "magnesium": 10, "alkalinity": 100,
             "minimum_run_volume": 300},
        ],
    }
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    blend = {a["source_id"]: a["volume"] for a in body["blend"]}
    assert blend["A"] == pytest.approx(400.0, abs=1e-6)
    assert blend["B"] == pytest.approx(300.0, abs=1e-6)


def test_solve_minimum_active_sources_infeasible(client):
    # Only A declares a threshold; K=2 can never be met even though B is used.
    payload = _commission_request(minimum_active_sources=2)
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    assert body["achievable_ranges"]["active_sources"] == {
        "requested": 2, "available": 1
    }
    assert "minimum_active_sources=2" in body["message"]


def test_minimum_run_infeasible_ranges_honor_constraint(client):
    # A ca100 threshold400 cap500, B ca20 cap300, V500: A must run >= 400,
    # achievable calcium [84, 100]; asking for <= 60 is infeasible.
    payload = {
        "target_volume": 500,
        "calcium": {"min": 10, "max": 60},
        "sources": [
            {"id": "A", "max_volume": 500, "cost": 1, "calcium": 100,
             "magnesium": 10, "alkalinity": 100,
             "minimum_run_volume": 400},
            {"id": "B", "max_volume": 300, "cost": 2, "calcium": 20,
             "magnesium": 10, "alkalinity": 100},
        ],
    }
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    ca = body["achievable_ranges"]["calcium"]
    assert ca["min"] == pytest.approx(84.0, abs=1e-6)
    assert ca["max"] == pytest.approx(100.0, abs=1e-6)


@pytest.mark.parametrize("bad", [0, -1, -0.5, "abc", True, False])
def test_minimum_run_volume_validation(client, bad):
    payload = make_request()
    payload["sources"][0]["minimum_run_volume"] = bad
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_minimum_run_volume_must_be_finite(client, token):
    raw = (
        '{"target_volume": 1000, "sources": ['
        '{"id": "W1", "max_volume": 600, "cost": 1.0, "calcium": 100,'
        ' "magnesium": 30, "alkalinity": 250, "minimum_run_volume": ' + token + '},'
        '{"id": "W2", "max_volume": 800, "cost": 2.0, "calcium": 20,'
        ' "magnesium": 5, "alkalinity": 80}'
        ']}'
    )
    r = client.post(
        "/api/v1/blends/solve", content=raw,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422, r.text


def test_minimum_run_volume_null_means_unset(client):
    payload = make_request()
    payload["sources"][0]["minimum_run_volume"] = None
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("bad", [-1, "abc", True, False, 1.5])
def test_minimum_active_sources_validation(client, bad):
    payload = make_request()
    payload["minimum_active_sources"] = bad
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 422, r.text


def test_minimum_run_part_of_idempotency_fingerprint(client):
    with_threshold = _commission_request(idempotency_key="comm-fp")
    r1 = client.post("/api/v1/blends", json=with_threshold)
    assert r1.status_code == 201
    # same key, minimum_run_volume dropped: different request -> conflict
    without_threshold = _commission_request(idempotency_key="comm-fp")
    without_threshold["sources"][0].pop("minimum_run_volume")
    r2 = client.post("/api/v1/blends", json=without_threshold)
    assert r2.status_code == 409


def test_minimum_active_sources_part_of_idempotency_fingerprint(client):
    with_k = {
        "target_volume": 700,
        "minimum_active_sources": 2,
        "idempotency_key": "comm-fp-k",
        "sources": [
            {"id": "A", "max_volume": 700, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100,
             "minimum_run_volume": 300},
            {"id": "B", "max_volume": 700, "cost": 2, "calcium": 50,
             "magnesium": 10, "alkalinity": 100,
             "minimum_run_volume": 300},
        ],
    }
    r1 = client.post("/api/v1/blends", json=with_k)
    assert r1.status_code == 201
    without_k = dict(with_k)
    without_k.pop("minimum_active_sources")
    r2 = client.post("/api/v1/blends", json=without_k)
    assert r2.status_code == 409


def test_commission_plan_persisted_and_replayed(client):
    payload = _commission_request(idempotency_key="comm-plan-1")
    r1 = client.post("/api/v1/blends", json=payload)
    assert r1.status_code == 201
    assert r1.json()["status"] == "optimal"
    r2 = client.post("/api/v1/blends", json=payload)
    assert r2.status_code == 200
    assert r2.headers.get("x-idempotent-replay") == "true"
    assert r2.json() == r1.json()
    g = client.get(f"/api/v1/blends/{r1.json()['plan_id']}")
    assert g.status_code == 200
    assert g.json() == r1.json()


def test_infeasible_commission_plan_is_persisted(client):
    payload = _commission_request(
        idempotency_key="comm-infeasible", minimum_active_sources=2
    )
    r = client.post("/api/v1/blends", json=payload)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "infeasible"
    assert body["achievable_ranges"]["active_sources"] == {
        "requested": 2, "available": 1
    }
    g = client.get(f"/api/v1/blends/{body['plan_id']}")
    assert g.status_code == 200
    assert g.json()["status"] == "infeasible"
    assert g.json()["achievable_ranges"]["active_sources"] == {
        "requested": 2, "available": 1
    }


def _save_commission_plan(client, key="comm-cont-1"):
    # V=700, K=2, A/B thresholds 300; cost optimum A=400, B=300.
    payload = {
        "target_volume": 700,
        "minimum_active_sources": 2,
        "idempotency_key": key,
        "sources": [
            {"id": "A", "max_volume": 700, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100,
             "minimum_run_volume": 300},
            {"id": "B", "max_volume": 700, "cost": 2, "calcium": 50,
             "magnesium": 10, "alkalinity": 100,
             "minimum_run_volume": 300},
        ],
    }
    r = client.post("/api/v1/blends", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "optimal"
    return body


def test_contingency_preserves_commissioning_constraints(client):
    plan = _save_commission_plan(client)
    # cut A to 350: both sources must stay active (K=2), A in [300, 350];
    # minimum change keeps A=350 and shifts 50 to B.
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"A": 350}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    blend = {a["source_id"]: a["new_volume"] for a in body["blend"]}
    assert blend["A"] == pytest.approx(350.0, abs=1e-6)
    assert blend["B"] == pytest.approx(350.0, abs=1e-6)
    assert body["total_adjustment"] == pytest.approx(100.0, abs=1e-6)


def test_contingency_commissioning_infeasible(client):
    plan = _save_commission_plan(client, key="comm-cont-inf")
    # cutting A to 100 makes its 300 threshold unreachable, so only one
    # threshold source can ever be active and K=2 cannot be met
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"A": 100}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    assert body["achievable_ranges"]["active_sources"] == {
        "requested": 2, "available": 1
    }
    assert "minimum_active_sources=2" in body["message"]
    # the stored plan is untouched
    assert client.get(
        f"/api/v1/blends/{plan['plan_id']}"
    ).json()["status"] == "optimal"


def test_contingency_minimum_run_shuts_source_off(client):
    # V=700, A threshold 300 cap 500, B continuous cap 1000; saved plan
    # A=500, B=200.  Cutting A to 250 disables it; B covers all 700.
    payload = {
        "target_volume": 700,
        "idempotency_key": "comm-cont-off",
        "sources": [
            {"id": "A", "max_volume": 500, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100,
             "minimum_run_volume": 300},
            {"id": "B", "max_volume": 1000, "cost": 2, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
        ],
    }
    plan = client.post("/api/v1/blends", json=payload).json()
    assert plan["status"] == "optimal"
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"A": 250}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    blend = {a["source_id"]: a["new_volume"] for a in body["blend"]}
    assert blend["A"] == pytest.approx(0.0, abs=1e-6)
    assert blend["B"] == pytest.approx(700.0, abs=1e-6)
