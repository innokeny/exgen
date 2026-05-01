def test_health_returns_status(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "loading"}
    assert body["default_model"] == "qwen2.5-3b"
    assert "qwen2.5-3b" in body["loaded_models"]
    assert "gpu" in body
    assert "available" in body["gpu"]
