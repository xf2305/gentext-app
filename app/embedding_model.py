"""Word embeddings with spaCy (from Module 2 Practical 3 - Word Embeddings)."""

import spacy


class EmbeddingModel:
    def __init__(self, model_name="en_core_web_lg"):
        # Loading takes a few seconds, so it happens once when the API starts.
        self.nlp = spacy.load(model_name)

    def calculate_embedding(self, input_word):
        """Return the embedding vector (list of floats) for a word or phrase.

        For more than one word, spaCy averages the word vectors.
        Words the model doesn't know get a vector of all zeros.
        """
        doc = self.nlp(input_word)
        return {
            "vector": doc.vector.tolist(),
            "has_vector": bool(doc.has_vector and doc.vector_norm > 0),
        }

    def calculate_similarity(self, word1, word2):
        """Cosine similarity between the embeddings of two words."""
        return float(self.nlp(word1).similarity(self.nlp(word2)))
