from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    assert client.get("/").json() == {"Hello": "World"}


def test_generate():
    r = client.post("/generate", json={"start_word": "bigram", "length": 5})
    assert r.status_code == 200
    assert r.json()["generated_text"].startswith("bigram")


def test_embedding_post():
    r = client.post("/embedding", json={"word": "apple"})
    assert r.status_code == 200
    body = r.json()
    assert body["word"] == "apple"
    assert body["dimension"] == 300
    assert len(body["embedding"]) == 300


def test_embedding_truncate_and_get():
    r = client.get("/embedding/orange?truncate=10")
    assert r.status_code == 200
    assert len(r.json()["embedding"]) == 10


def test_embedding_unknown_word():
    r = client.post("/embedding", json={"word": "qwzxkvbnmplk"})
    assert r.status_code == 404


def test_similarity():
    r = client.post("/similarity", json={"word1": "apple", "word2": "orange"})
    assert r.status_code == 200
    assert 0 < r.json()["similarity"] <= 1
