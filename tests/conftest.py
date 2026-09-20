import os
import tempfile

# Point the service at a throwaway database before importing the app.
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="blend-test-"), "test.db")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Fresh database file per test; the lifespan handler creates the schema.
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def make_request(**overrides):
    payload = {
        "target_volume": 1000,
        "calcium": {"min": 40, "max": 60},
        "sources": [
            {
                "id": "W1",
                "max_volume": 600,
                "cost": 1.0,
                "calcium": 100,
                "magnesium": 30,
                "alkalinity": 250,
            },
            {
                "id": "W2",
                "max_volume": 800,
                "cost": 2.0,
                "calcium": 20,
                "magnesium": 5,
                "alkalinity": 80,
            },
        ],
    }
    payload.update(overrides)
    return payload
