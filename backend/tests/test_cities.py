"""Station registry: CRUD contract, validation, and idempotent startup seeding."""

import pytest

from app.core.cities import DEFAULT_CITIES
from app.repositories.city_repo import seed_default_cities


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
        json={"name": "  Sevilla ", "latitude": 37.3891, "longitude": -5.9845},
    )

    assert response.status_code == 201
    assert response.json() == {"name": "Sevilla", "latitude": 37.3891, "longitude": -5.9845}
    names = [city["name"] for city in (await client.get("/api/cities")).json()]
    assert names == ["Alicante", "Madrid", "Sevilla"]


async def test_create_duplicate_is_case_insensitive_409(client):
    response = await client.post(
        "/api/cities", json={"name": "madrid", "latitude": 1.0, "longitude": 2.0}
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "City 'madrid' already exists"}


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
    response = await client.post("/api/cities", json=payload)

    assert response.status_code == 400
    assert "detail" in response.json()


async def test_delete_city_is_204_and_registry_only(client):
    response = await client.delete("/api/cities/Madrid")

    assert response.status_code == 204
    assert response.content == b""
    assert (await client.get("/api/cities")).json() == [
        {"name": "Alicante", "latitude": 38.3452, "longitude": -0.481}
    ]


async def test_delete_unknown_city_is_404(client):
    response = await client.delete("/api/cities/Atlantis")

    assert response.status_code == 404
    assert response.json() == {"detail": "City 'Atlantis' not found"}


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
    assert len(DEFAULT_CITIES) == 14
    assert ("Malaga", 36.7213, -4.4214) in DEFAULT_CITIES  # unaccented on purpose


async def test_seed_inserts_defaults_when_empty():
    collection = _FakeCollection()

    seeded = await seed_default_cities(_FakeDb(collection))

    assert seeded is True
    assert collection.inserted is not None
    assert len(collection.inserted) == 14
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
