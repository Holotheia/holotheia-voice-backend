#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIVING ORCHESTRATOR — Orchestrateur vivant pipeline complet avec Claude

Architecture:
- Intégration tous composants (brain, fusion, anti-rigid, vector, guards)
- Pipeline query → routes → sélection → exécution → validation
- LLM integration (Claude Anthropic) pour génération réponse finale
- Evolution continue via mutations
- Historique conversationnel

Principe:
Orchestre cycle complet: query → recherche modules → génération routes →
sélection meilleure route → exécution → validation guards → génération Claude

Date: 2025-12-08
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

from holotheia_core.claude_connector import ClaudeConnector


class LivingOrchestrator:
    """
    Orchestrateur vivant — Pipeline complet auto-évolutif

    Coordonne tous composants pour traiter queries avec évolution continue.
    """

    def __init__(
        self,
        brain,
        fusion_engine,
        anti_rigid,
        vector_store,
        guards,
        anthropic_api_key: Optional[str] = None
    ):
        """
        Initialise orchestrateur

        Args:
            brain: FractalBrain instance
            fusion_engine: MorphicFusionEngine instance
            anti_rigid: AntiRigidificationEngine instance
            vector_store: HolotheiaVectorStore instance
            guards: HolotheiaGuards instance
            anthropic_api_key: Clé API Anthropic (optionnel)
        """
        self.brain = brain
        self.fusion_engine = fusion_engine
        self.anti_rigid = anti_rigid
        self.vector_store = vector_store
        self.guards = guards

        # Claude connector
        self.claude = ClaudeConnector(api_key=anthropic_api_key)
        self.llm_enabled = self.claude.enabled

        # Historique conversation
        self.conversation_history: List[Dict] = []

        print(f"🎭 LivingOrchestrator initialized (LLM: {'Claude' if self.llm_enabled else 'mock mode'})")

    def process_query(
        self,
        query: str,
        max_routes: int = 10,
        force_innovation: bool = False
    ) -> Dict:
        """
        Traite query complète

        Pipeline:
        1. Check innovation forcée
        2. Recherche modules pertinents (vector store)
        3. Génération routes (fusion engine)
        4. Sélection meilleure route
        5. Exécution route (activation modules + fusion)
        6. Génération réponse (Claude)
        7. Validation (guards)
        8. Evolution (anti-cristallisation)

        Args:
            query: Requête utilisateur
            max_routes: Nombre max routes à considérer
            force_innovation: Force innovation même si non détecté

        Returns:
            Résultat complet avec réponse, traces, validations
        """
        start_time = datetime.utcnow()

        result = {
            'query': query,
            'timestamp': start_time.isoformat(),
            'pipeline_steps': [],
            'response': None,
            'validation': None,
            'evolution': None,
            'error': None
        }

        try:
            # STEP 1: Check innovation forcée
            should_innovate = force_innovation or self.anti_rigid.should_force_innovation(query)

            if should_innovate:
                innovation = self.anti_rigid.force_innovation(reason='query_triggered')
                result['pipeline_steps'].append({
                    'step': 'forced_innovation',
                    'innovation': innovation
                })

            # STEP 2: Recherche modules (vector store)
            vector_results = self.vector_store.search_modules(query, k=30)

            result['pipeline_steps'].append({
                'step': 'vector_search',
                'results_count': len(vector_results)
            })

            # STEP 3: Génération routes (fusion engine)
            routes = self.fusion_engine.generate_all_possible_routes(
                query,
                max_depth=5,
                min_relevance=0.1
            )

            result['pipeline_steps'].append({
                'step': 'route_generation',
                'routes_count': len(routes)
            })

            if not routes:
                # Aucune route → création module émergent
                new_module = self._create_emergent_module(query)
                result['pipeline_steps'].append({
                    'step': 'emergent_module_creation',
                    'module': new_module
                })

                # Re-génération routes
                routes = self.fusion_engine.generate_all_possible_routes(query, max_depth=3)

            # STEP 4: Sélection meilleure route
            best_route = routes[0] if routes else None

            if best_route:
                result['pipeline_steps'].append({
                    'step': 'route_selection',
                    'route': {
                        'type': best_route['type'],
                        'depth': best_route['depth'],
                        'score': best_route['score'],
                        'description': best_route['description']
                    }
                })

                # STEP 5: Exécution route
                execution = self.fusion_engine.execute_route(
                    best_route,
                    context={'query': query, 'timestamp': start_time.isoformat()}
                )

                result['pipeline_steps'].append({
                    'step': 'route_execution',
                    'execution': {
                        'fusion_created': execution['fusion_created'],
                        'modules_activated': len(execution['modules_activated'])
                    }
                })

                # STEP 6: Génération réponse (Claude)
                response_text = self.claude.generate_response(
                    query=query,
                    route=best_route,
                    execution=execution,
                    conversation_history=self.conversation_history
                )

                result['response'] = response_text

                result['pipeline_steps'].append({
                    'step': 'response_generation',
                    'method': 'claude' if self.llm_enabled else 'mock'
                })

                # STEP 7: Validation (guards)
                validation = self.guards.validate_response(
                    response_text,
                    modules_used=best_route['modules'],
                    history=[h['response'] for h in self.conversation_history[-5:]]
                )

                result['validation'] = validation

                result['pipeline_steps'].append({
                    'step': 'response_validation',
                    'is_valid': validation['is_valid'],
                    'alerts_count': len(validation['alerts'])
                })

                # STEP 8: Evolution (anti-cristallisation)
                evolution_report = self.anti_rigid.apply_anti_crystallization()

                result['evolution'] = {
                    'crystallization_detected': evolution_report['diagnosis']['is_crystallized'],
                    'interventions_count': len(evolution_report['interventions'])
                }

                result['pipeline_steps'].append({
                    'step': 'system_evolution',
                    'evolution': result['evolution']
                })

            else:
                result['error'] = 'No routes generated'

            # Update conversation history
            self.conversation_history.append({
                'query': query,
                'response': result.get('response'),
                'timestamp': start_time.isoformat(),
                'valid': result.get('validation', {}).get('is_valid', False)
            })

            # Update vector store
            self.vector_store.update_from_brain(self.brain)

        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()

        # Durée totale
        end_time = datetime.utcnow()
        result['duration_ms'] = (end_time - start_time).total_seconds() * 1000

        return result

    def _create_emergent_module(self, query: str) -> Dict:
        """
        Crée module émergent depuis query

        Args:
            query: Requête

        Returns:
            Module créé
        """
        module = self.brain.create_module(
            name=f"emergent_{hash(query) % 10000}",
            description=f"Emerged from query: {query[:50]}",
            module_type="emergent_concept",
            context={'query': query, 'emergent': True}
        )

        # Ajout au vector store
        self.vector_store.add_module(module)

        return module

    def get_brain_status(self) -> Dict:
        """Retourne statut complet système"""
        return {
            'brain': self.brain.get_brain_status(),
            'vector_store': self.vector_store.get_stats(),
            'anti_rigid': self.anti_rigid.get_innovation_stats(),
            'guards': self.guards.get_guard_stats(),
            'conversation_history_size': len(self.conversation_history),
            'llm_enabled': self.llm_enabled
        }

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Retourne historique conversation"""
        return self.conversation_history[-limit:]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎭 LIVING ORCHESTRATOR — TEST (Claude)")
    print("=" * 70)

    # Import composants
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from holotheia_core.fractal_brain import FractalBrain
    from holotheia_core.morphic_fusion_engine import MorphicFusionEngine
    from holotheia_core.anti_rigidification import AntiRigidificationEngine
    from holotheia_core.vector_store import HolotheiaVectorStore
    from holotheia_core.guards import HolotheiaGuards

    # Création composants
    print("\n[1] Initialisation composants...")
    brain = FractalBrain(brain_path="./test_brain_orchestrator_claude")
    fusion_engine = MorphicFusionEngine(brain)
    anti_rigid = AntiRigidificationEngine(brain, innovation_probability=0.2)
    vector_store = HolotheiaVectorStore(persist_directory="./test_chroma_orch_claude")
    guards = HolotheiaGuards()

    print("✓ Composants créés")

    # Création modules initiaux
    print("\n[2] Création modules initiaux...")
    modules_init = [
        ("semantic_search", "Recherche sémantique vectorielle", "function"),
        ("morphic_field", "Champ morphique résonance", "concept"),
        ("fusion_engine", "Moteur fusion conceptuelle", "algorithm")
    ]

    for name, desc, mtype in modules_init:
        m = brain.create_module(name, desc, mtype)
        brain.activate_module(m['id'])
        vector_store.add_module(m)
        print(f"✓ Module: {name}")

    # Création orchestrateur
    print("\n[3] Création orchestrateur avec Claude...")
    orchestrator = LivingOrchestrator(
        brain=brain,
        fusion_engine=fusion_engine,
        anti_rigid=anti_rigid,
        vector_store=vector_store,
        guards=guards,
        anthropic_api_key=None  # Mock mode pour test
    )
    print("✓ Orchestrateur créé")

    # Test query
    print("\n[4] Test query: 'recherche morphique'...")
    result = orchestrator.process_query("recherche morphique", max_routes=5)

    print(f"   Durée: {result['duration_ms']:.2f}ms")
    print(f"   Steps: {len(result['pipeline_steps'])}")
    print(f"   Response: {result['response'][:100] if result['response'] else 'None'}...")
    print(f"   Valid: {result.get('validation', {}).get('is_valid', 'N/A')}")
    print(f"   Error: {result.get('error', 'None')}")

    # Statut système
    print("\n[5] Statut système...")
    status = orchestrator.get_brain_status()
    print(f"   Modules: {status['brain']['nb_modules']}")
    print(f"   LLM: {'Claude' if status['llm_enabled'] else 'Mock'}")

    print("\n✅ Test terminé — Living Orchestrator avec Claude opérationnel")
    print("=" * 70)
