"""Lifespan wiring: ping before serving, indexes/seed in the background, close on shutdown."""

import asyncio
import logging

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


async def test_lifespan_pings_before_serving_then_prepares_in_background(settings, monkeypatch):
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
        # Mongo is pinged before the first request; indexes/seed are not.
        assert calls == ["ping"]
        task = app.state.prepare_database_task
        assert task is not None
        await task
        assert calls == ["ping", "indexes", "seed"]
        assert app.state.mongo_client is client
        assert app.state.db is client.database
        assert app.state.aemet.configured is False
        assert app.state.geo is not None

    assert client.closed is True


async def test_lifespan_can_skip_startup_preparation(settings, monkeypatch):
    client = _FakeClient()
    calls: list[str] = []

    async def fake_ping(_client):
        calls.append("ping")

    async def fake_ensure_indexes(db):  # pragma: no cover - must not run
        calls.append("indexes")

    async def fake_seed_cities(db):  # pragma: no cover - must not run
        calls.append("seed")
        return True

    monkeypatch.setattr(main_module, "create_client", lambda _settings: client)
    monkeypatch.setattr(main_module, "ping", fake_ping)
    monkeypatch.setattr(main_module, "ensure_indexes", fake_ensure_indexes)
    monkeypatch.setattr(main_module, "seed_default_cities", fake_seed_cities)

    disabled = settings.model_copy(update={"ensure_indexes_on_start": False})
    app = main_module.create_app(disabled)

    async with app.router.lifespan_context(app):
        assert calls == ["ping"]
        assert app.state.prepare_database_task is None

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


async def test_lifespan_shutdown_bounds_and_cancels_stuck_preparation(
    settings, monkeypatch, caplog
):
    client = _FakeClient()

    async def fake_ping(_client):
        return None

    async def stuck_ensure_indexes(_db):
        await asyncio.Event().wait()  # never completes

    monkeypatch.setattr(main_module, "create_client", lambda _settings: client)
    monkeypatch.setattr(main_module, "ping", fake_ping)
    monkeypatch.setattr(main_module, "ensure_indexes", stuck_ensure_indexes)
    monkeypatch.setattr(main_module, "seed_default_cities", lambda _db: asyncio.sleep(0))
    monkeypatch.setattr(main_module, "PREPARE_SHUTDOWN_TIMEOUT_SECONDS", 0.05)

    app = main_module.create_app(settings)

    with caplog.at_level(logging.WARNING):
        async with app.router.lifespan_context(app):
            pass  # shutdown must not block on the stuck index task

    assert "did not finish within" in caplog.text
    assert client.closed is True
