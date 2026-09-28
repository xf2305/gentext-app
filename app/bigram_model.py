"""Bigram text generator (from Module 2 Practical 2 - Word Sampling)."""

import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        """Build bigram probabilities from a list of strings (or one string)."""
        text = " ".join(corpus) if isinstance(corpus, (list, tuple)) else corpus
        self.vocab, self.bigram_probs = self.analyze_bigrams(text, frequency_threshold)

    @staticmethod
    def simple_tokenizer(text, frequency_threshold=None):
        """Lowercase the text and split it into words."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not frequency_threshold:
            return tokens
        word_counts = Counter(tokens)
        return [t for t in tokens if word_counts[t] >= frequency_threshold]

    def analyze_bigrams(self, text, frequency_threshold=None):
        """p(w2 | w1) = C(w1, w2) / C(w1)"""
        words = self.simple_tokenizer(text, frequency_threshold)
        bigram_counts = Counter(zip(words[:-1], words[1:]))
        unigram_counts = Counter(words)

        bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]
        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word, length=20):
        """Sample `length` words, each one chosen from p(next | current)."""
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:  # no known continuation -> stop
                break
            current_word = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            generated_words.append(current_word)

        return " ".join(generated_words)
