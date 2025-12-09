#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VECTOR STORE — Mémoire vectorielle sémantique avec ChromaDB

Architecture:
- Stockage embeddings modules/fusions
- Recherche sémantique vectorielle
- Intégration SentenceTransformer
- Persistance ChromaDB
- Queries par similarité cosine

Principe:
Transforme modules/fusions en vecteurs sémantiques pour permettre
recherche par similarité au lieu de simple matching mots-clés.

Date: 2025-12-06
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime


class HolotheiaVectorStore:
    """
    Mémoire vectorielle — Recherche sémantique modules/fusions

    Note: Implémentation simplifiée sans dépendances externes.
    En production, intégrer ChromaDB + SentenceTransformer.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "holotheia_modules"
    ):
        """
        Initialise vector store

        Args:
            persist_directory: Chemin persistance ChromaDB
            collection_name: Nom collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Stockage simplifié (mock ChromaDB)
        # En production: self.client = chromadb.Client(...)
        self.documents: Dict[str, Dict] = {}
        self.embeddings: Dict[str, List[float]] = {}

        print(f"📦 VectorStore initialized (mock mode)")

    def _compute_simple_embedding(self, text: str) -> List[float]:
        """
        Calcule embedding simplifié (mock)

        En production: utiliser SentenceTransformer
        embeddings = self.encoder.encode([text])

        Args:
            text: Texte à encoder

        Returns:
            Vecteur embedding (mock)
        """
        # Mock: hash text vers vecteur 384-dim (taille all-MiniLM-L6-v2)
        # En production: return self.encoder.encode(text).tolist()

        # Simple hash-based mock
        h = hash(text)
        vec = []
        for i in range(384):
            val = ((h + i * 31) % 1000) / 1000.0
            vec.append(val)

        return vec

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcule similarité cosine entre 2 vecteurs"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def add_module(self, module: Dict):
        """
        Ajoute module au vector store

        Args:
            module: Module avec keys: id, name, description, type
        """
        module_id = module['id']

        # Texte pour embedding
        text = f"{module['name']} {module['description']} {module['type']}"

        # Compute embedding
        embedding = self._compute_simple_embedding(text)

        # Stockage
        self.documents[module_id] = {
            'id': module_id,
            'text': text,
            'metadata': {
                'name': module['name'],
                'type': module['type'],
                'created_at': module.get('created_at', datetime.utcnow().isoformat())
            }
        }

        self.embeddings[module_id] = embedding

    def add_fusion(self, fusion: Dict):
        """
        Ajoute fusion au vector store

        Args:
            fusion: Fusion avec keys: id, description, module_names
        """
        fusion_id = fusion['id']

        # Texte pour embedding
        text = f"{fusion['description']} {' '.join(fusion['module_names'])}"

        # Compute embedding
        embedding = self._compute_simple_embedding(text)

        # Stockage
        self.documents[fusion_id] = {
            'id': fusion_id,
            'text': text,
            'metadata': {
                'description': fusion['description'],
                'type': 'fusion',
                'module_count': len(fusion['module_ids']),
                'created_at': fusion.get('created_at', datetime.utcnow().isoformat())
            }
        }

        self.embeddings[fusion_id] = embedding

    def search_modules(
        self,
        query: str,
        k: int = 30,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Recherche modules/fusions par similarité sémantique

        Args:
            query: Requête texte
            k: Nombre résultats max
            min_score: Score minimum

        Returns:
            Liste modules/fusions avec scores
        """
        if not self.embeddings:
            return []

        # Embedding requête
        query_embedding = self._compute_simple_embedding(query)

        # Calcul similarités
        results = []

        for doc_id, doc_embedding in self.embeddings.items():
            score = self._cosine_similarity(query_embedding, doc_embedding)

            if score >= min_score:
                results.append({
                    'id': doc_id,
                    'score': score,
                    'document': self.documents[doc_id]
                })

        # Tri par score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:k]

    def get_similar_modules(
        self,
        module_id: str,
        k: int = 10
    ) -> List[Dict]:
        """
        Trouve modules similaires à un module donné

        Args:
            module_id: ID module source
            k: Nombre résultats

        Returns:
            Modules similaires
        """
        if module_id not in self.embeddings:
            return []

        source_embedding = self.embeddings[module_id]

        # Calcul similarités
        results = []

        for doc_id, doc_embedding in self.embeddings.items():
            if doc_id == module_id:
                continue  # Skip self

            score = self._cosine_similarity(source_embedding, doc_embedding)

            results.append({
                'id': doc_id,
                'score': score,
                'document': self.documents[doc_id]
            })

        # Tri par score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:k]

    def update_from_brain(self, brain):
        """
        Met à jour vector store depuis FractalBrain

        Args:
            brain: Instance FractalBrain
        """
        # Ajout modules
        for module in brain.modules.values():
            if module['id'] not in self.documents:
                self.add_module(module)

        # Ajout fusions
        for fusion in brain.fusions.values():
            if fusion['id'] not in self.documents:
                self.add_fusion(fusion)

        print(f"✓ VectorStore updated: {len(self.documents)} documents")

    def get_stats(self) -> Dict:
        """Retourne statistiques vector store"""
        types = {}
        for doc in self.documents.values():
            dtype = doc['metadata'].get('type', 'module')
            types[dtype] = types.get(dtype, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_embeddings': len(self.embeddings),
            'types': types,
            'embedding_dim': len(next(iter(self.embeddings.values()))) if self.embeddings else 0
        }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("📦 VECTOR STORE — TEST")
    print("=" * 70)

    # Import cerveau
    from fractal_brain import FractalBrain

    # Création cerveau + modules
    brain = FractalBrain(brain_path="./test_brain_vector")

    print("\n[1] Création modules test...")
    modules_test = [
        ("semantic_search", "Recherche sémantique vectorielle", "function"),
        ("morphic_resonance", "Résonance morphique champ", "concept"),
        ("fusion_engine", "Moteur fusion conceptuelle", "algorithm"),
        ("vector_embedding", "Embedding vectoriel dense", "function"),
        ("neural_network", "Réseau neuronal transformeur", "algorithm")
    ]

    module_ids = []
    for name, desc, mtype in modules_test:
        m = brain.create_module(name, desc, mtype)
        module_ids.append(m['id'])
        print(f"✓ Module créé: {name}")

    # Création fusion
    print("\n[2] Création fusion...")
    fusion = brain.create_fusion(
        module_ids[:3],
        fusion_type="semantic",
        description="Fusion recherche sémantique morphique"
    )
    print(f"✓ Fusion créée: {fusion['description']}")

    # Création vector store
    print("\n[3] Initialisation vector store...")
    vector_store = HolotheiaVectorStore(persist_directory="./test_chroma")
    print("✓ VectorStore initialisé")

    # Update depuis cerveau
    print("\n[4] Update depuis cerveau...")
    vector_store.update_from_brain(brain)

    # Recherche sémantique
    print("\n[5] Recherche sémantique: 'neural embedding'...")
    results = vector_store.search_modules("neural embedding", k=5)

    print(f"✓ {len(results)} résultats:")
    for i, result in enumerate(results, 1):
        doc = result['document']
        print(f"\n   #{i} (score: {result['score']:.4f})")
        print(f"      {doc['metadata'].get('name', 'N/A')}")
        print(f"      Type: {doc['metadata'].get('type', 'N/A')}")

    # Modules similaires
    print("\n[6] Modules similaires à 'semantic_search'...")
    similar = vector_store.get_similar_modules(module_ids[0], k=3)

    print(f"✓ {len(similar)} résultats:")
    for i, result in enumerate(similar, 1):
        doc = result['document']
        print(f"   #{i} (score: {result['score']:.4f}) - {doc['metadata'].get('name', 'N/A')}")

    # Statistiques
    print("\n[7] Statistiques vector store...")
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✅ Test terminé — Vector store opérationnel")
    print("=" * 70)
