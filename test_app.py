import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json["project"] == "AutoStockML"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"


def test_valid_prediction(client):
    response = client.post(
        "/predict",
        json={"prices": [100, 101, 102, 101.5, 103]},
    )
    assert response.status_code == 200
    assert response.json["model"] == "LinearRegression"


@pytest.mark.parametrize("prices", [
    [100, 101],
    [100, 101, 102, 103, 104, 105],
    [100, -1, 102, 103, 104],
    [100, "bad", 102, 103, 104],
    [100, float("nan"), 102, 103, 104],
])
def test_invalid_prices(client, prices):
    response = client.post("/predict", json={"prices": prices})
    assert response.status_code == 400


def test_missing_prices(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_invalid_json(client):
    response = client.post(
        "/predict",
        data="not json",
        content_type="application/json",
    )
    assert response.status_code == 400