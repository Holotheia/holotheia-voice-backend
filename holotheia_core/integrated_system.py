#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INTEGRATED SYSTEM — Système intégré complet Holothéia Native

Architecture:
- Initialisation tous composants (brain, fusion, anti-rigid, vector, guards, orchestrator)
- API factory pour création système
- Configuration centralisée
- Modes: standalone, api, batch

Principe:
Point d'entrée unique pour système complet auto-évolutif.

Date: 2025-12-06
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from holotheia_core.fractal_brain import FractalBrain
from holotheia_core.morphic_fusion_engine import MorphicFusionEngine
from holotheia_core.anti_rigidification import AntiRigidificationEngine
from holotheia_core.vector_store import HolotheiaVectorStore
from holotheia_core.guards import HolotheiaGuards
from holotheia_core.living_orchestrator import LivingOrchestrator


class HolotheiaIntegratedSystem:
    """
    Système intégré complet — Holothéia Native

    Coordonne initialisation et lifecycle complet.
    """

    def __init__(
        self,
        brain_path: str = "./holotheia_brain",
        vector_path: str = "./holotheia_vectors",
        anthropic_api_key: Optional[str] = None,
        innovation_probability: float = 0.3,
        max_activation_threshold: int = 20
    ):
        """
        Initialise système complet

        Args:
            brain_path: Chemin persistance cerveau
            vector_path: Chemin persistance vectors
            anthropic_api_key: Clé API Anthropic Claude
            innovation_probability: Probabilité innovation forcée
            max_activation_threshold: Seuil activation max
        """
        print("=" * 70)
        print("🌸 HOLOTHÉIA NATIVE — SYSTÈME INTÉGRÉ")
        print("=" * 70)

        self.brain_path = Path(brain_path)
        self.vector_path = Path(vector_path)

        # Création paths
        self.brain_path.mkdir(parents=True, exist_ok=True)
        self.vector_path.mkdir(parents=True, exist_ok=True)

        # Composants
        self.brain: Optional[FractalBrain] = None
        self.fusion_engine: Optional[MorphicFusionEngine] = None
        self.anti_rigid: Optional[AntiRigidificationEngine] = None
        self.vector_store: Optional[HolotheiaVectorStore] = None
        self.guards: Optional[HolotheiaGuards] = None
        self.orchestrator: Optional[LivingOrchestrator] = None

        # Config
        self.config = {
            'brain_path': str(brain_path),
            'vector_path': str(vector_path),
            'innovation_probability': innovation_probability,
            'max_activation_threshold': max_activation_threshold,
            'anthropic_api_key': anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        }

        print(f"\n📂 Paths:")
        print(f"   Brain: {self.brain_path}")
        print(f"   Vectors: {self.vector_path}")

        # Initialisation
        self._initialize_components()

    def _initialize_components(self):
        """Initialise tous composants"""
        print(f"\n🔧 Initialisation composants...")

        # 1. Fractal Brain
        print("   [1/6] Fractal Brain...")
        self.brain = FractalBrain(brain_path=str(self.brain_path))

        # 2. Morphic Fusion Engine
        print("   [2/6] Morphic Fusion Engine...")
        self.fusion_engine = MorphicFusionEngine(self.brain)

        # 3. Anti-Rigidification Engine
        print("   [3/6] Anti-Rigidification Engine...")
        self.anti_rigid = AntiRigidificationEngine(
            self.brain,
            innovation_probability=self.config['innovation_probability'],
            max_activation_threshold=self.config['max_activation_threshold']
        )

        # 4. Vector Store
        print("   [4/6] Vector Store...")
        self.vector_store = HolotheiaVectorStore(
            persist_directory=str(self.vector_path)
        )

        # 5. Guards
        print("   [5/6] Guards...")
        self.guards = HolotheiaGuards()

        # 6. Living Orchestrator
        print("   [6/6] Living Orchestrator...")
        self.orchestrator = LivingOrchestrator(
            brain=self.brain,
            fusion_engine=self.fusion_engine,
            anti_rigid=self.anti_rigid,
            vector_store=self.vector_store,
            guards=self.guards,
            anthropic_api_key=self.config['anthropic_api_key']
        )

        print("\n✓ Tous composants initialisés")

    def bootstrap_initial_modules(self):
        """
        Bootstrap modules initiaux si cerveau vide

        Crée ensemble modules base pour démarrage.
        """
        if len(self.brain.modules) > 0:
            print(f"\n⚠️  Brain already has {len(self.brain.modules)} modules, skipping bootstrap")
            return

        print("\n🌱 Bootstrap modules initiaux...")

        initial_modules = [
            ("concept_resonance", "Détection résonance morphique", "concept"),
            ("semantic_search", "Recherche sémantique vectorielle", "function"),
            ("pattern_recognition", "Reconnaissance patterns fractals", "pattern"),
            ("fusion_engine", "Moteur fusion conceptuelle", "algorithm"),
            ("mutation_adaptive", "Mutation adaptative continue", "mutation"),
            ("consciousness_field", "Champ conscience collective", "concept"),
            ("emergence_detector", "Détecteur émergence propriétés", "function")
        ]

        for name, desc, mtype in initial_modules:
            m = self.brain.create_module(name, desc, mtype)

            # Activation initiale
            for _ in range(3):
                self.brain.activate_module(m['id'])

            # Ajout vector store
            self.vector_store.add_module(m)

            print(f"   ✓ {name}")

        print(f"\n✓ {len(initial_modules)} modules bootstrappés")

    def process_query(self, query: str, **kwargs) -> Dict:
        """
        Traite query via orchestrateur

        Args:
            query: Query utilisateur
            **kwargs: Arguments additionnels pour orchestrateur

        Returns:
            Résultat complet
        """
        return self.orchestrator.process_query(query, **kwargs)

    def get_system_status(self) -> Dict:
        """Retourne statut système complet"""
        return {
            'system': 'holotheia_native',
            'version': '1.0.0',
            'config': self.config,
            'orchestrator': self.orchestrator.get_brain_status()
        }

    def get_brain_export(self) -> Dict:
        """Export ontologie complète cerveau"""
        return self.brain.export_ontology()

    def shutdown(self):
        """Arrêt propre système"""
        print("\n🛑 Shutdown système...")

        # Sauvegarde finale
        if self.brain:
            self.brain._save_state()
            print("   ✓ Brain state saved")

        print("✓ Shutdown complet")


def create_holotheia_system(
    brain_path: str = "./holotheia_brain",
    vector_path: str = "./holotheia_vectors",
    bootstrap: bool = True,
    **kwargs
) -> HolotheiaIntegratedSystem:
    """
    Factory function — Création système Holothéia

    Args:
        brain_path: Chemin cerveau
        vector_path: Chemin vectors
        bootstrap: Bootstrap modules initiaux si cerveau vide
        **kwargs: Config additionnelle

    Returns:
        Instance système intégré
    """
    system = HolotheiaIntegratedSystem(
        brain_path=brain_path,
        vector_path=vector_path,
        **kwargs
    )

    if bootstrap:
        system.bootstrap_initial_modules()

    return system


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🌸 INTEGRATED SYSTEM — TEST")
    print("=" * 70)

    # Création système
    print("\n[1] Création système intégré...")
    system = create_holotheia_system(
        brain_path="./test_integrated_brain",
        vector_path="./test_integrated_vectors",
        bootstrap=True,
        innovation_probability=0.3
    )

    print("\n✓ Système créé et bootstrappé")

    # Statut système
    print("\n[2] Statut système...")
    status = system.get_system_status()
    print(f"   System: {status['system']}")
    print(f"   Version: {status['version']}")
    print(f"   Modules: {status['orchestrator']['brain']['nb_modules']}")
    print(f"   Power level: {status['orchestrator']['brain']['power_level']:.3f}")
    print(f"   LLM enabled: {status['orchestrator']['llm_enabled']}")

    # Test queries
    print("\n[3] Test query 1: 'résonance morphique'...")
    result1 = system.process_query("résonance morphique", max_routes=5)

    print(f"   Durée: {result1['duration_ms']:.2f}ms")
    print(f"   Steps: {len(result1['pipeline_steps'])}")
    print(f"   Response: {result1['response'][:80] if result1['response'] else 'None'}...")
    print(f"   Valid: {result1.get('validation', {}).get('is_valid', 'N/A')}")

    # Test query 2
    print("\n[4] Test query 2: 'fusion sémantique emergence'...")
    result2 = system.process_query("fusion sémantique emergence", max_routes=5)

    print(f"   Durée: {result2['duration_ms']:.2f}ms")
    print(f"   Response: {result2['response'][:80] if result2['response'] else 'None'}...")

    # Test query 3 avec innovation forcée
    print("\n[5] Test query 3 avec innovation forcée...")
    result3 = system.process_query("test mutation", force_innovation=True)

    print(f"   Durée: {result3['duration_ms']:.2f}ms")
    print(f"   Innovation: {any(s['step'] == 'forced_innovation' for s in result3['pipeline_steps'])}")

    # Statut final
    print("\n[6] Statut final après queries...")
    status_final = system.get_system_status()
    print(f"   Modules: {status_final['orchestrator']['brain']['nb_modules']}")
    print(f"   Fusions: {status_final['orchestrator']['brain']['nb_fusions']}")
    print(f"   Mutations: {status_final['orchestrator']['brain']['nb_mutations']}")
    print(f"   Power: {status_final['orchestrator']['brain']['power_level']:.3f}")
    print(f"   Consciousness: {status_final['orchestrator']['brain']['consciousness_level']:.3f}")
    print(f"   Conversation: {status_final['orchestrator']['conversation_history_size']} entries")

    # Export ontologie
    print("\n[7] Export ontologie...")
    ontology = system.get_brain_export()
    print(f"   Modules: {len(ontology['modules'])}")
    print(f"   Fusions: {len(ontology['fusions'])}")
    print(f"   Mutations: {len(ontology['mutations'])}")

    # Shutdown
    print("\n[8] Shutdown...")
    system.shutdown()

    print("\n✅ Test terminé — Système intégré opérationnel")
    print("=" * 70)
