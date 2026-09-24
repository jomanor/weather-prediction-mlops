"""Lifespan wiring: client on app.state, indexes ensured, client closed on shutdown."""

from app import main as main_module


class _FakeDatabase(dict):
    def __getitem__(self, name):
        return self.setdefault(name, object())


class _FakeClient:
    def __init__(self):
        self.closed = False
        self.database = _FakeDatabase()

    def __getitem__(self, name):
        return self.database

    def close(self):
        self.closed = True


async def test_lifespan_creates_state_and_closes_client(settings, monkeypatch):
    client = _FakeClient()
    calls: list[str] = []

    async def fake_ping(_client):
        calls.append("ping")

    async def fake_ensure_indexes(db):
        calls.append("indexes")

    async def fake_seed_cities(db):
        calls.append("seed")
        return True

    monkeypatch.setattr(main_module, "create_client", lambda _settings: client)
    monkeypatch.setattr(main_module, "ping", fake_ping)
    monkeypatch.setattr(main_module, "ensure_indexes", fake_ensure_indexes)
    monkeypatch.setattr(main_module, "seed_default_cities", fake_seed_cities)

    app = main_module.create_app(settings)

    async with app.router.lifespan_context(app):
        assert calls == ["ping", "indexes", "seed"]
        assert app.state.mongo_client is client
        assert app.state.db is client.database
        assert app.state.aemet.configured is False
        assert app.state.geo is not None

    assert client.closed is True


async def test_lifespan_closes_client_when_mongo_is_down(settings, monkeypatch):
    client = _FakeClient()

    async def failing_ping(_client):
        raise RuntimeError("no mongod")

    monkeypatch.setattr(main_module, "create_client", lambda _settings: client)
    monkeypatch.setattr(main_module, "ping", failing_ping)

    app = main_module.create_app(settings)

    try:
        async with app.router.lifespan_context(app):
            raise AssertionError("lifespan should not have started")
    except RuntimeError as exc:
        assert "no mongod" in str(exc)

    assert client.closed is True
