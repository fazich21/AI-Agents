"""
Entity Extractor: Scans chunks and extracts named entities
using spaCy's NER (Named Entity Recognition).

Entities include:
  - PERSON: people's names
  - ORG: organizations, companies
  - GPE: countries, cities
  - TECH: technologies (custom patterns)
  - TOPIC: AI/ML topics (custom patterns)

Uses spaCy locally — no API calls, no Gemini quota used.
"""

import re
import json
import spacy
from collections import defaultdict
from config import CHUNKS_PATH, DATA_DIR
import os


# ──────────────────────────────────────────────
# Custom tech/topic patterns that spaCy misses
# ──────────────────────────────────────────────

TECH_PATTERNS = [
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "neural networks", "convolutional neural network",
    "recurrent neural network", "natural language processing",
    "computer vision", "reinforcement learning", "supervised learning",
    "unsupervised learning", "transfer learning", "backpropagation",
    "gradient descent", "random forest", "decision tree", "support vector machine",
    "generative adversarial network", "transformer", "attention mechanism",
    "bert", "gpt", "lstm", "cnn", "rnn", "gan",
    "autonomous vehicle", "self-driving", "robotics",
    "image recognition", "speech recognition", "object detection",
    "sentiment analysis", "recommendation system",
    "tensorflow", "pytorch", "keras", "scikit-learn",
    "chatbot", "virtual assistant", "expert system",
    "knowledge graph", "ontology", "semantic web",
    "big data", "data mining", "feature extraction",
    "classification", "regression", "clustering",
    "overfitting", "underfitting", "regularization",
    "activation function", "loss function", "optimizer",
    "batch normalization", "dropout", "embedding",
    "künstliche intelligenz", "maschinelles lernen",
    "tiefes lernen", "neuronales netz",
    "مصنوعی ذہانت", "مشین لرننگ", "ڈیپ لرننگ",
]

TOPIC_PATTERNS = [
    "healthcare", "medical imaging", "drug discovery", "diagnosis",
    "education system", "automotive industry", "self-driving car",
    "climate change", "global warming", "renewable energy",
    "cybersecurity", "privacy", "ethics",
    "bildungssystem", "automobilindustrie", "medizin",
    "gesundheitswesen", "elektromobilität",
]


class EntityExtractor:
    """Extract entities from text chunks using spaCy + custom patterns."""

    def __init__(self):
        """Load spaCy model."""
        print("  Loading spaCy NER model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("  Downloading spaCy model (one-time, ~12MB)...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        print("  Entity extractor ready!")

        # Build pattern lookup (lowercase for matching)
        self.tech_patterns = set(p.lower() for p in TECH_PATTERNS)
        self.topic_patterns = set(p.lower() for p in TOPIC_PATTERNS)

    def extract_entities(self, text):
        """
        Extract entities from a text string.

        Returns: list of dicts with {name, type, start, end}
        """
        entities = []
        seen = set()

        # 1. spaCy NER for people, orgs, locations
        doc = self.nlp(text[:5000])  # Limit length for speed
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "NORP", "FAC", "EVENT"):
                name = ent.text.strip()
                if len(name) > 2 and name.lower() not in seen:
                    seen.add(name.lower())
                    entities.append({
                        "name": name,
                        "type": ent.label_,
                    })

        # 2. Custom pattern matching for tech terms
        text_lower = text.lower()
        for pattern in self.tech_patterns:
            if pattern in text_lower and pattern not in seen:
                seen.add(pattern)
                entities.append({
                    "name": pattern.title() if len(pattern) > 3 else pattern.upper(),
                    "type": "TECH",
                })

        # 3. Custom pattern matching for topics
        for pattern in self.topic_patterns:
            if pattern in text_lower and pattern not in seen:
                seen.add(pattern)
                entities.append({
                    "name": pattern.title(),
                    "type": "TOPIC",
                })

        return entities

    def extract_from_all_chunks(self, chunks=None):
        """
        Extract entities from all chunks.

        Returns:
            entity_index: {entity_name: {type, chunk_ids: [...]}}
            chunk_entities: {chunk_id: [entity_names]}
        """
        if chunks is None:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)

        print(f"  Extracting entities from {len(chunks)} chunks...")

        entity_index = {}       # entity_name -> {type, chunk_ids}
        chunk_entities = {}     # chunk_id -> [entity_names]

        for i, chunk in enumerate(chunks):
            if i % 500 == 0:
                print(f"    Processing chunk {i}/{len(chunks)}...")

            entities = self.extract_entities(chunk["text"])
            chunk_id = chunk["chunk_id"]
            chunk_entities[chunk_id] = []

            for ent in entities:
                name = ent["name"]
                ent_type = ent["type"]

                if name not in entity_index:
                    entity_index[name] = {
                        "type": ent_type,
                        "chunk_ids": [],
                        "count": 0,
                    }

                entity_index[name]["chunk_ids"].append(chunk_id)
                entity_index[name]["count"] += 1
                chunk_entities[chunk_id].append(name)

        # Filter: keep only entities that appear in at least 2 chunks
        entity_index = {
            name: info for name, info in entity_index.items()
            if info["count"] >= 2
        }

        print(f"  Extracted {len(entity_index)} unique entities (appearing 2+ times)")

        # Show top entities
        top = sorted(entity_index.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        print(f"  Top entities:")
        for name, info in top:
            print(f"    {name} ({info['type']}): {info['count']} chunks")

        return entity_index, chunk_entities


# ──────────────────────────────────────────────
# Test it
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ENTITY EXTRACTOR TEST")
    print("=" * 60)

    extractor = EntityExtractor()

    # Test on a sample text
    sample = """
    Deep learning is a subset of machine learning that uses neural networks 
    with many layers. Google and Microsoft have invested heavily in AI research. 
    Backpropagation is the key algorithm for training neural networks using 
    gradient descent. In Germany, the automotive industry is adopting AI 
    for autonomous vehicles.
    """

    entities = extractor.extract_entities(sample)
    print(f"\n  Sample text entities:")
    for ent in entities:
        print(f"    {ent['name']} ({ent['type']})")

    # Test on full corpus
    print(f"\n  Full corpus extraction:")
    entity_index, chunk_entities = extractor.extract_from_all_chunks()

    # Save
    save_path = os.path.join(DATA_DIR, "entities.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(entity_index, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved entities to {save_path}")
