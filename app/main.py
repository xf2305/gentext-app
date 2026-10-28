from fastapi import FastAPI, HTTPException, File, UploadFile, Query, Response
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.cnn_classifier import CNNClassifier
from app.digit_generator import DigitGenerator

app = FastAPI(title="Gentext API", description="Bigram text generation and spaCy word embeddings")
cnn_classifier = CNNClassifier()
digit_generator = DigitGenerator()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()


# ---------- Request models ----------
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, examples=["apple"])
    # Optional: return only the first N numbers of the vector (like `[:10]` in the notebook)
    truncate: int | None = Field(None, ge=1, le=300, examples=[10])


class SimilarityRequest(BaseModel):
    word1: str = Field(..., min_length=1, examples=["apple"])
    word2: str = Field(..., min_length=1, examples=["orange"])


# ---------- Endpoints ----------
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    """Return the spaCy word embedding for the query word."""
    word = request.word.strip()
    if not word:
        raise HTTPException(status_code=422, detail="word must not be blank")

    result = embedding_model.calculate_embedding(word)
    if not result["has_vector"]:
        raise HTTPException(status_code=404, detail=f"No embedding found for '{word}'")

    vector = result["vector"]
    return {
        "word": word,
        "dimension": len(vector),
        "embedding": vector[: request.truncate] if request.truncate else vector,
    }


@app.get("/embedding/{word}")
def get_embedding_by_path(word: str, truncate: int | None = None):
    """Same as POST /embedding, but you can call it from a browser: /embedding/apple"""
    return get_embedding(EmbeddingRequest(word=word, truncate=truncate))


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    """Bonus: cosine similarity between two words' embeddings."""
    return {
        "word1": request.word1,
        "word2": request.word2,
        "similarity": embedding_model.calculate_similarity(request.word1, request.word2),
    }
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Assignment 2: classify an uploaded image into one of the 10 CIFAR-10 classes."""
    image_bytes = await file.read()
    try:
        result = cnn_classifier.predict(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file as an image")
    return {"filename": file.filename, **result}
@app.get("/generate_digits")
def generate_digits(n: int = Query(16, ge=1, le=64)):
    """Assignment 3: generate n handwritten digits with the trained GAN, returned as a PNG grid."""
    image_bytes = digit_generator.generate(n)
    return Response(content=image_bytes, media_type="image/png")