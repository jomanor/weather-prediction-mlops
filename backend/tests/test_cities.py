"""Station registry: CRUD contract, validation, write gating, and seeding."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.cities import DEFAULT_CITIES
from app.core.config import Settings
from app.main import create_app
from app.repositories.city_repo import get_city_repo, seed_default_cities
from tests.conftest import CITY_WRITES_TOKEN

#: Credentials the ``settings`` fixture accepts for a registry write.
WRITE_AUTH = {"Authorization": f"Bearer {CITY_WRITES_TOKEN}"}
#: Both prefixes serve the same route objects; the gate must cover each mount.
WRITE_PREFIXES = ["/api/cities", "/api/v1/cities"]


async def test_list_cities_returns_objects_sorted(client):
    response = await client.get("/api/cities")

    assert response.status_code == 200
    assert response.json() == [
        {"name": "Alicante", "latitude": 38.3452, "longitude": -0.481},
        {"name": "Madrid", "latitude": 40.4168, "longitude": -3.7038},
    ]


async def test_create_city_trims_name_and_persists(client):
    response = await client.post(
        "/api/cities",
        headers=WRITE_AUTH,
        json={"name": "  Sevilla ", "latitude": 37.3891, "longitude": -5.9845},
    )

    assert response.status_code == 201
    assert response.json() == {"name": "Sevilla", "latitude": 37.3891, "longitude": -5.9845}
    names = [city["name"] for city in (await client.get("/api/cities")).json()]
    assert names == ["Alicante", "Madrid", "Sevilla"]


async def test_create_duplicate_is_case_insensitive_409(client):
    response = await client.post(
        "/api/cities",
        headers=WRITE_AUTH,
        json={"name": "madrid", "latitude": 1.0, "longitude": 2.0},
    )

    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "conflict"
    assert body["error"]["message"] == "City 'madrid' already exists"
    assert body["error"]["request_id"] == response.headers["x-request-id"]


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "X", "latitude": 90.1, "longitude": 0.0},
        {"name": "X", "latitude": -90.1, "longitude": 0.0},
        {"name": "X", "latitude": 0.0, "longitude": 180.1},
        {"name": "X", "latitude": 0.0, "longitude": -180.1},
        {"name": "   ", "latitude": 0.0, "longitude": 0.0},
        {"name": "X" * 101, "latitude": 0.0, "longitude": 0.0},
    ],
)
async def test_create_invalid_payload_is_400(client, payload):
    response = await client.post("/api/cities", headers=WRITE_AUTH, json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


async def test_delete_city_is_204_and_registry_only(client):
    response = await client.delete("/api/cities/Madrid", headers=WRITE_AUTH)

    assert response.status_code == 204
    assert response.content == b""
    assert (await client.get("/api/cities")).json() == [
        {"name": "Alicante", "latitude": 38.3452, "longitude": -0.481}
    ]


async def test_delete_unknown_city_is_404(client):
    response = await client.delete("/api/cities/Atlantis", headers=WRITE_AUTH)

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "City 'Atlantis' not found"


# ---------------------------------------------------------------------------
# Write gating (CITY_WRITES_TOKEN)
# ---------------------------------------------------------------------------


def _unconfigured_client(city_repo) -> AsyncClient:
    """An app whose settings leave ``city_writes_token`` unset (the shipped default)."""
    application = create_app(Settings(_env_file=None))
    application.dependency_overrides[get_city_repo] = lambda: city_repo
    return AsyncClient(transport=ASGITransport(app=application), base_url="http://test")


@pytest.mark.parametrize("prefix", WRITE_PREFIXES)
async def test_writes_are_503_when_no_token_is_configured(prefix, city_repo):
    new_city = {"name": "Sevilla", "latitude": 37.3891, "longitude": -5.9845}
    async with _unconfigured_client(city_repo) as unconfigured:
        created = await unconfigured.post(prefix, headers=WRITE_AUTH, json=new_city)
        deleted = await unconfigured.delete(f"{prefix}/Madrid", headers=WRITE_AUTH)

    for response in (created, deleted):
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "city_writes_disabled"
        assert error["message"] == "City writes are disabled"
        assert error["request_id"] == response.headers["x-request-id"]
    # The gate runs before the handler: no read, no write ever reached the repo.
    assert city_repo.calls == []
    assert [city.name for city in city_repo.cities] == ["Madrid", "Alicante"]


@pytest.mark.parametrize("prefix", WRITE_PREFIXES)
async def test_reads_are_unaffected_when_no_token_is_configured(prefix, city_repo):
    async with _unconfigured_client(city_repo) as unconfigured:
        response = await unconfigured.get(prefix)

    assert response.status_code == 200
    assert [city["name"] for city in response.json()] == ["Alicante", "Madrid"]


@pytest.mark.parametrize("prefix", WRITE_PREFIXES)
@pytest.mark.parametrize(
    "headers",
    [
        {},  # no header at all
        {"Authorization": "Bearer wrong-token"},  # wrong secret
        {"Authorization": f"Bearer {CITY_WRITES_TOKEN}x"},  # prefix of the right one
        {"Authorization": CITY_WRITES_TOKEN},  # right secret, wrong scheme
        {"Authorization": "Basic dXNlcjpwdw=="},
    ],
)
async def test_bad_token_is_401_on_both_writes(prefix, headers, client, city_repo):
    new_city = {"name": "Sevilla", "latitude": 37.3891, "longitude": -5.9845}

    created = await client.post(prefix, headers=headers, json=new_city)
    deleted = await client.delete(f"{prefix}/Madrid", headers=headers)

    for response in (created, deleted):
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        error = response.json()["error"]
        assert error["code"] == "unauthorized"
        assert error["request_id"] == response.headers["x-request-id"]
        # The submitted value is never echoed back.
        assert "Bearer" not in error["message"]
        assert CITY_WRITES_TOKEN not in response.text
    assert city_repo.calls == []
    assert [city.name for city in city_repo.cities] == ["Madrid", "Alicante"]


@pytest.mark.parametrize("prefix", WRITE_PREFIXES)
async def test_correct_token_allows_both_writes(prefix, client, city_repo):
    created = await client.post(
        prefix,
        headers=WRITE_AUTH,
        json={"name": "Sevilla", "latitude": 37.3891, "longitude": -5.9845},
    )
    deleted = await client.delete(f"{prefix}/Madrid", headers=WRITE_AUTH)

    assert created.status_code == 201
    assert created.json() == {"name": "Sevilla", "latitude": 37.3891, "longitude": -5.9845}
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert city_repo.calls == ["get", "upsert", "delete"]
    assert [city.name for city in city_repo.cities] == ["Alicante", "Sevilla"]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self, docs=None, fail=False):
        self.docs = list(docs or [])
        self.fail = fail
        self.inserted: list[dict] | None = None

    async def count_documents(self, _filter):
        if self.fail:
            raise RuntimeError("no mongod")
        return len(self.docs)

    async def insert_many(self, documents, ordered=False):
        self.inserted = list(documents)
        self.docs.extend(documents)


class _FakeDb:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, _name):
        return self.collection


def test_default_cities_keep_the_producer_names_and_coordinates():
    assert len(DEFAULT_CITIES) == 15
    assert ("Malaga", 36.7213, -4.4214) in DEFAULT_CITIES  # unaccented on purpose
    # Pasadena is the only non-Spanish station (Open-Meteo producer, no AEMET).
    assert ("Pasadena", 34.1478, -118.1445) in DEFAULT_CITIES


async def test_seed_inserts_defaults_when_empty():
    collection = _FakeCollection()

    seeded = await seed_default_cities(_FakeDb(collection))

    assert seeded is True
    assert collection.inserted is not None
    assert len(collection.inserted) == 15
    assert collection.inserted[0] == {
        "name": "El Ejido",
        "name_key": "el ejido",
        "latitude": 36.7756,
        "longitude": -2.8144,
    }


async def test_seed_is_idempotent_when_collection_is_not_empty():
    collection = _FakeCollection(
        [{"name": "Madrid", "name_key": "madrid", "latitude": 40.4, "longitude": -3.7}]
    )

    seeded = await seed_default_cities(_FakeDb(collection))

    assert seeded is False
    assert collection.inserted is None


async def test_seed_swallows_mongo_failure():
    collection = _FakeCollection(fail=True)

    seeded = await seed_default_cities(_FakeDb(collection))

    assert seeded is False
