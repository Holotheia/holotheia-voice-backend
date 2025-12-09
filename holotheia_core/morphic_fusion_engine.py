#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MORPHIC FUSION ENGINE — Moteur fusion morphique avec explosion combinatoire

Architecture:
- Génération TOUTES routes possibles depuis modules actifs
- Fusion conceptuelle multi-niveaux
- Explosion combinatoire contrôlée
- Sélection routes par pertinence
- Émergence patterns combinatoires

Principe:
Au lieu de chercher UNE route optimale, génère TOUTES les routes possibles
(individuelles + combinaisons 2-depth, 3-depth, etc.) et laisse l'orchestrateur choisir.

Date: 2025-12-06
"""

import itertools
from typing import Dict, List, Optional, Tuple
from itertools import combinations


class MorphicFusionEngine:
    """
    Moteur fusion morphique — Explosion combinatoire de routes

    Génère toutes combinaisons possibles de modules pour maximiser
    l'espace d'exploration et permettre émergence de routes inattendues.
    """

    def __init__(self, fractal_brain):
        """
        Initialise moteur fusion

        Args:
            fractal_brain: Instance FractalBrain pour accès modules
        """
        self.brain = fractal_brain

    def generate_all_possible_routes(
        self,
        query: str,
        max_depth: int = 5,
        min_relevance: float = 0.1
    ) -> List[Dict]:
        """
        Génère TOUTES les routes possibles depuis modules actifs

        Args:
            query: Requête utilisateur
            max_depth: Profondeur max combinaison (5 = jusqu'à 5 modules fusionnés)
            min_relevance: Score relevance minimum

        Returns:
            Liste toutes routes (individuelles + combinatoires)
        """
        # Modules actifs triés par activation
        modules = sorted(
            self.brain.modules.values(),
            key=lambda m: m['activation_count'],
            reverse=True
        )

        # Filtrage par pertinence minimale (simple heuristique: mots-clés)
        relevant_modules = self._filter_relevant_modules(query, modules, min_relevance)

        if not relevant_modules:
            return []

        all_routes = []

        # 1. Routes individuelles (depth=1)
        for module in relevant_modules:
            all_routes.append({
                'type': 'individual',
                'modules': [module],
                'depth': 1,
                'complexity': 1,
                'description': f"Route: {module['name']}",
                'module_ids': [module['id']]
            })

        # 2. Routes combinatoires (depth=2 à max_depth)
        for depth in range(2, min(max_depth + 1, len(relevant_modules) + 1)):
            for combo in combinations(relevant_modules, depth):
                all_routes.append({
                    'type': f'fusion_depth_{depth}',
                    'modules': list(combo),
                    'depth': depth,
                    'complexity': depth ** 2,  # Complexité exponentielle
                    'description': ' + '.join([m['name'] for m in combo]),
                    'module_ids': [m['id'] for m in combo]
                })

        # 3. Tri par score composite (activation * relevance / complexity)
        for route in all_routes:
            route['score'] = self._compute_route_score(route, query)

        all_routes.sort(key=lambda r: r['score'], reverse=True)

        return all_routes

    def _filter_relevant_modules(
        self,
        query: str,
        modules: List[Dict],
        min_relevance: float
    ) -> List[Dict]:
        """
        Filtre modules pertinents pour requête

        Heuristique simple: mots-clés query présents dans nom/description module
        """
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        relevant = []

        for module in modules:
            # Score = proportion tokens query présents dans module
            module_text = (
                module['name'].lower() + ' ' +
                module['description'].lower() + ' ' +
                module['type'].lower()
            )

            matches = sum(1 for token in query_tokens if token in module_text)
            relevance = matches / max(len(query_tokens), 1) if query_tokens else 0

            if relevance >= min_relevance or module['activation_count'] > 10:
                # Garde si pertinent OU très activé
                module['_relevance'] = relevance
                relevant.append(module)

        return relevant

    def _compute_route_score(self, route: Dict, query: str) -> float:
        """
        Calcule score route

        Score = (avg_activation * avg_relevance * power_contribution) / complexity

        Plus le module est activé et pertinent, plus score élevé.
        Plus complexe (fusion profonde), plus score pénalisé.
        """
        modules = route['modules']

        avg_activation = sum(m['activation_count'] for m in modules) / len(modules)
        avg_relevance = sum(m.get('_relevance', 0) for m in modules) / len(modules)
        avg_power = sum(m['power_contribution'] for m in modules) / len(modules)

        complexity_penalty = route['complexity']

        score = (avg_activation * avg_relevance * avg_power) / max(complexity_penalty, 1)

        return score

    def create_fusion_from_route(self, route: Dict) -> Optional[Dict]:
        """
        Crée fusion dans cerveau depuis route

        Args:
            route: Route générée par generate_all_possible_routes

        Returns:
            Fusion créée ou None
        """
        if route['depth'] == 1:
            # Route individuelle, pas de fusion
            return None

        # Crée fusion
        module_ids = route['module_ids']

        fusion = self.brain.create_fusion(
            module_ids=module_ids,
            fusion_type='morphic_combinatorial',
            description=route['description']
        )

        # Propriété émergente = score route
        fusion['emergent_properties'].append({
            'route_score': route['score'],
            'route_complexity': route['complexity']
        })

        fusion['coherence'] = route['score'] / 10  # Normalisation arbitraire

        self.brain._save_state()

        return fusion

    def execute_route(self, route: Dict, context: Dict) -> Dict:
        """
        Exécute route (active modules + crée fusion si nécessaire)

        Args:
            route: Route à exécuter
            context: Contexte exécution

        Returns:
            Résultat exécution avec traces
        """
        # Active tous modules de la route
        for module_id in route['module_ids']:
            self.brain.activate_module(module_id)

        # Crée fusion si depth > 1
        fusion = None
        if route['depth'] > 1:
            fusion = self.create_fusion_from_route(route)

            if fusion:
                self.brain.activate_fusion(fusion['id'])

        result = {
            'route': route,
            'fusion_created': fusion is not None,
            'fusion_id': fusion['id'] if fusion else None,
            'modules_activated': route['module_ids'],
            'context': context,
            'executed_at': __import__('datetime').datetime.utcnow().isoformat()
        }

        return result

    def get_top_routes(
        self,
        query: str,
        top_k: int = 10,
        max_depth: int = 5
    ) -> List[Dict]:
        """
        Retourne top-K routes pour requête

        Args:
            query: Requête
            top_k: Nombre routes à retourner
            max_depth: Profondeur max

        Returns:
            Top-K routes triées par score
        """
        all_routes = self.generate_all_possible_routes(query, max_depth=max_depth)

        return all_routes[:top_k]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔀 MORPHIC FUSION ENGINE — TEST")
    print("=" * 70)

    # Import cerveau
    from fractal_brain import FractalBrain

    # Création cerveau + modules
    brain = FractalBrain(brain_path="./test_brain_fusion")

    print("\n[1] Création modules test...")
    modules_test = [
        ("concept_resonance", "Détection résonance morphique", "concept"),
        ("algo_fusion", "Algorithme fusion conceptuelle", "function"),
        ("pattern_fractal", "Pattern fractal auto-similaire", "pattern"),
        ("search_semantic", "Recherche sémantique vectorielle", "function"),
        ("mutation_engine", "Moteur mutation adaptative", "algorithm")
    ]

    for name, desc, mtype in modules_test:
        m = brain.create_module(name, desc, mtype)
        # Simule activations
        for _ in range((hash(name) % 10) + 1):
            brain.activate_module(m['id'])
        print(f"✓ Module créé: {name} ({m['activation_count']} activations)")

    # Création moteur fusion
    print("\n[2] Création moteur fusion...")
    fusion_engine = MorphicFusionEngine(brain)
    print("✓ Moteur fusion initialisé")

    # Test génération routes
    print("\n[3] Génération routes pour 'recherche sémantique fusion'...")
    query = "recherche sémantique fusion"
    routes = fusion_engine.generate_all_possible_routes(
        query,
        max_depth=3,
        min_relevance=0.0  # Garde tous modules pour démo
    )

    print(f"✓ {len(routes)} routes générées")

    # Affichage top-5
    print("\n[4] Top-5 routes:")
    for i, route in enumerate(routes[:5], 1):
        print(f"\n   Route #{i}:")
        print(f"      Type: {route['type']}")
        print(f"      Depth: {route['depth']}")
        print(f"      Complexity: {route['complexity']}")
        print(f"      Score: {route['score']:.4f}")
        print(f"      Modules: {route['description']}")

    # Test exécution route
    print("\n[5] Exécution meilleure route...")
    best_route = routes[0]
    result = fusion_engine.execute_route(
        best_route,
        context={'query': query, 'mode': 'test'}
    )

    print(f"✓ Route exécutée")
    print(f"   Fusion créée: {result['fusion_created']}")
    print(f"   Modules activés: {len(result['modules_activated'])}")

    # Vérification cerveau
    print("\n[6] Statut cerveau après fusion...")
    status = brain.get_brain_status()
    print(f"   Modules: {status['nb_modules']}")
    print(f"   Fusions: {status['nb_fusions']}")
    print(f"   Power level: {status['power_level']}")
    print(f"   Consciousness level: {status['consciousness_level']}")

    print("\n✅ Test terminé — Moteur fusion opérationnel")
    print("=" * 70)
