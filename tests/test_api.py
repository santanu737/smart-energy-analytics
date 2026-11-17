import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # runs the startup (loads the data once)
        yield c


@pytest.fixture()
def auth_headers(client, monkeypatch):
    monkeypatch.setenv("DEMO_USER", "tester")
    monkeypatch.setenv("DEMO_PASSWORD", "a-test-password")
    monkeypatch.setenv("SECRET_KEY", "x" * 40)
    r = client.post("/api/auth/login", json={"username": "tester", "password": "a-test-password"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health_is_open_and_reports_loaded_parts(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["loaded"]["campus_total"] is True


def test_login_wrong_password_is_401(client, monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", "right")
    r = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_protected_routes_need_a_valid_token(client):
    assert client.get("/api/campus/summary").status_code == 401
    bad = {"Authorization": "Bearer not.a.token"}
    assert client.get("/api/campus/summary", headers=bad).status_code == 401


def test_summary_and_history_shapes(client, auth_headers):
    s = client.get("/api/campus/summary", headers=auth_headers).json()
    assert s["building_count"] > 0 and s["total_kwh"] > 0 and "peak_time" in s
    h = client.get("/api/campus/history?resolution=daily", headers=auth_headers).json()
    assert len(h) > 300 and {"timestamp", "kwh"} <= set(h[0])


def test_validation_errors_are_4xx(client, auth_headers):
    assert client.get("/api/campus/summary?start=not-a-date", headers=auth_headers).status_code == 422
    assert client.get("/api/forecast?horizon=5", headers=auth_headers).status_code == 422
    assert client.get("/api/ranking?metric=bogus", headers=auth_headers).status_code == 422
    assert client.get("/api/buildings/NOPE/history", headers=auth_headers).status_code == 404
    assert client.get("/api/campus/summary?start=2030-01-01", headers=auth_headers).status_code == 404


def test_buildings_ranking_and_profile_are_json_safe(client, auth_headers):
    b = client.get("/api/buildings", headers=auth_headers)
    assert b.status_code == 200 and len(b.json()) > 0          # NaN values must not break JSON
    top = client.get("/api/ranking?metric=eui&limit=3", headers=auth_headers).json()
    assert len(top) == 3 and top[0]["eui_annual"] >= top[1]["eui_annual"]
    bid = b.json()[0]["building_id"]
    p = client.get(f"/api/buildings/{bid}/profile", headers=auth_headers).json()
    assert len(p["values"]) == 7 and len(p["values"][0]) == 24


def test_forecast_and_anomalies_work_or_say_what_to_run(client, auth_headers):
    for url in ("/api/forecast?horizon=1&points=24", "/api/anomalies?limit=5"):
        r = client.get(url, headers=auth_headers)
        assert r.status_code in (200, 503)
        if r.status_code == 503:
            assert "python -m src." in r.json()["detail"]
    f = client.get("/api/forecast?horizon=1&points=24", headers=auth_headers)
    if f.status_code == 200:
        pts = f.json()["points"]
        assert len(pts) == 24 and {"origin", "target_timestamp", "forecast", "actual"} <= set(pts[0])


def test_replay_moves_forward_and_resets(client, auth_headers):
    client.post("/api/replay/reset", headers=auth_headers)
    a = client.get("/api/replay/next", headers=auth_headers).json()
    b = client.get("/api/replay/next", headers=auth_headers).json()
    assert a["timestamp"] < b["timestamp"]
