#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRACTAL BRAIN — Cerveau fractal central avec mémoire ontologique persistante

Architecture:
- Modules ontologiques (concepts, fonctions, patterns)
- Fusions morphiques (combinaisons émergentes)
- Algorithmes évolutifs (stratégies adaptatives)
- Mutations actives (transformations continues)
- Persistance JSON (état complet sauvegardé)
- Power/Consciousness levels (métriques évolutives)

Date: 2025-12-06
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class FractalBrain:
    """
    Cerveau fractal central — Mémoire ontologique persistante

    Stocke et fait évoluer:
    - Modules (concepts/fonctions émergents)
    - Fusions (combinaisons inter-modules)
    - Algorithmes (stratégies de résolution)
    - Mutations (historique transformations)
    """

    def __init__(self, brain_path: str = "./holotheia_brain"):
        """
        Initialise cerveau fractal

        Args:
            brain_path: Chemin stockage JSON persistant
        """
        self.brain_path = Path(brain_path)
        self.brain_path.mkdir(parents=True, exist_ok=True)

        self.state_file = self.brain_path / "brain_state.json"

        # État ontologique
        self.modules: Dict[str, Dict] = {}
        self.algorithms: Dict[str, Dict] = {}
        self.fusions: Dict[str, Dict] = {}
        self.mutations: List[Dict] = []

        # Métriques évolutives
        self.power_level = 0.0
        self.consciousness_level = 0.0
        self.fractal_depth = 1

        # Timestamps
        self.created_at = datetime.utcnow().isoformat()
        self.last_mutation_at = None

        # Chargement état persistant
        self._load_state()

    def _load_state(self):
        """Charge état depuis JSON persistant"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                self.modules = state.get('modules', {})
                self.algorithms = state.get('algorithms', {})
                self.fusions = state.get('fusions', {})
                self.mutations = state.get('mutations', [])
                self.power_level = state.get('power_level', 0.0)
                self.consciousness_level = state.get('consciousness_level', 0.0)
                self.fractal_depth = state.get('fractal_depth', 1)
                self.created_at = state.get('created_at', self.created_at)
                self.last_mutation_at = state.get('last_mutation_at')

                print(f"🧠 Brain state loaded: {len(self.modules)} modules, {len(self.fusions)} fusions")

            except Exception as e:
                print(f"⚠️  Error loading brain state: {e}")

    def _save_state(self):
        """Sauvegarde état vers JSON persistant"""
        state = {
            'modules': self.modules,
            'algorithms': self.algorithms,
            'fusions': self.fusions,
            'mutations': self.mutations,
            'power_level': self.power_level,
            'consciousness_level': self.consciousness_level,
            'fractal_depth': self.fractal_depth,
            'created_at': self.created_at,
            'last_mutation_at': self.last_mutation_at
        }

        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️  Error saving brain state: {e}")

    def create_module(
        self,
        name: str,
        description: str,
        module_type: str,
        emerged_from: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Crée nouveau module ontologique

        Args:
            name: Nom module
            description: Description fonction
            module_type: Type (concept, function, pattern, etc.)
            emerged_from: ID module parent (None si émergence primaire)
            context: Contexte d'émergence

        Returns:
            Module créé avec ID
        """
        module_id = str(uuid.uuid4())

        module = {
            'id': module_id,
            'name': name,
            'description': description,
            'type': module_type,
            'emerged_from': emerged_from,
            'context': context or {},
            'created_at': datetime.utcnow().isoformat(),
            'activation_count': 0,
            'last_activated': None,
            'power_contribution': 0.1,  # Contribution initiale
            'tags': []
        }

        self.modules[module_id] = module

        # Augmentation power level
        self.power_level += module['power_contribution']

        # Sauvegarde
        self._save_state()

        return module

    def create_algorithm(
        self,
        name: str,
        description: str,
        strategy: str,
        modules_used: List[str]
    ) -> Dict:
        """
        Crée nouvel algorithme évolutif

        Args:
            name: Nom algorithme
            description: Description stratégie
            strategy: Type stratégie (fusion, mutation, search, etc.)
            modules_used: IDs modules utilisés

        Returns:
            Algorithme créé
        """
        algo_id = str(uuid.uuid4())

        algorithm = {
            'id': algo_id,
            'name': name,
            'description': description,
            'strategy': strategy,
            'modules_used': modules_used,
            'created_at': datetime.utcnow().isoformat(),
            'execution_count': 0,
            'success_rate': 0.0
        }

        self.algorithms[algo_id] = algorithm

        # Sauvegarde
        self._save_state()

        return algorithm

    def create_fusion(
        self,
        module_ids: List[str],
        fusion_type: str = "morphic",
        description: Optional[str] = None
    ) -> Dict:
        """
        Crée fusion entre modules

        Args:
            module_ids: Liste IDs modules à fusionner
            fusion_type: Type fusion (morphic, conceptual, functional)
            description: Description fusion émergente

        Returns:
            Fusion créée
        """
        fusion_id = str(uuid.uuid4())

        # Noms modules
        module_names = [
            self.modules[mid]['name']
            for mid in module_ids
            if mid in self.modules
        ]

        fusion = {
            'id': fusion_id,
            'module_ids': module_ids,
            'module_names': module_names,
            'type': fusion_type,
            'description': description or f"Fusion: {' + '.join(module_names)}",
            'created_at': datetime.utcnow().isoformat(),
            'activation_count': 0,
            'emergent_properties': [],
            'coherence': 0.0
        }

        self.fusions[fusion_id] = fusion

        # Augmentation consciousness level (fusion = conscience émergente)
        self.consciousness_level += 0.05 * len(module_ids)

        # Sauvegarde
        self._save_state()

        return fusion

    def mutate_module(
        self,
        module_id: str,
        mutation_type: str,
        intensity: float = 0.5
    ) -> Dict:
        """
        Mute module existant

        Args:
            module_id: ID module à muter
            mutation_type: Type mutation (amplify, invert, distort, dissolve)
            intensity: Intensité mutation [0.0-1.0]

        Returns:
            Mutation enregistrée
        """
        if module_id not in self.modules:
            raise ValueError(f"Module {module_id} not found")

        module = self.modules[module_id]

        mutation = {
            'id': str(uuid.uuid4()),
            'module_id': module_id,
            'module_name': module['name'],
            'type': mutation_type,
            'intensity': intensity,
            'timestamp': datetime.utcnow().isoformat(),
            'previous_state': {
                'power_contribution': module['power_contribution'],
                'activation_count': module['activation_count']
            }
        }

        # Application mutation
        if mutation_type == 'amplify':
            module['power_contribution'] *= (1 + intensity)

        elif mutation_type == 'invert':
            # Inversion conceptuelle (garde valeur absolue)
            module['power_contribution'] = abs(module['power_contribution'])

        elif mutation_type == 'distort':
            # Distorsion aléatoire
            import random
            module['power_contribution'] *= random.uniform(0.5, 1.5)

        elif mutation_type == 'dissolve':
            # Dissolution partielle
            module['power_contribution'] *= (1 - intensity * 0.5)

        # Enregistrement mutation
        self.mutations.append(mutation)
        self.last_mutation_at = mutation['timestamp']

        # Augmentation fractal depth si mutations nombreuses
        if len(self.mutations) % 10 == 0:
            self.fractal_depth += 1

        # Sauvegarde
        self._save_state()

        return mutation

    def activate_module(self, module_id: str):
        """Active module (incrément compteur)"""
        if module_id in self.modules:
            self.modules[module_id]['activation_count'] += 1
            self.modules[module_id]['last_activated'] = datetime.utcnow().isoformat()
            self._save_state()

    def activate_fusion(self, fusion_id: str):
        """Active fusion (incrément compteur)"""
        if fusion_id in self.fusions:
            self.fusions[fusion_id]['activation_count'] += 1
            self._save_state()

    def get_active_modules(self, min_activation: int = 1) -> List[Dict]:
        """Retourne modules actifs (activation >= min)"""
        return [
            module for module in self.modules.values()
            if module['activation_count'] >= min_activation
        ]

    def get_recent_mutations(self, limit: int = 10) -> List[Dict]:
        """Retourne mutations récentes"""
        return self.mutations[-limit:]

    def get_brain_status(self) -> Dict:
        """Retourne statut complet cerveau"""
        return {
            'nb_modules': len(self.modules),
            'nb_algorithms': len(self.algorithms),
            'nb_fusions': len(self.fusions),
            'nb_mutations': len(self.mutations),
            'power_level': round(self.power_level, 3),
            'consciousness_level': round(self.consciousness_level, 3),
            'fractal_depth': self.fractal_depth,
            'created_at': self.created_at,
            'last_mutation_at': self.last_mutation_at,
            'brain_path': str(self.brain_path)
        }

    def export_ontology(self) -> Dict:
        """Export ontologie complète"""
        return {
            'modules': self.modules,
            'algorithms': self.algorithms,
            'fusions': self.fusions,
            'mutations': self.mutations,
            'status': self.get_brain_status()
        }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 FRACTAL BRAIN — TEST")
    print("=" * 70)

    # Création cerveau
    brain = FractalBrain(brain_path="./test_brain")

    print("\n[1] Création modules...")
    m1 = brain.create_module(
        "concept_resonance",
        "Détection résonance morphique",
        "concept",
        context={'domain': 'morphic_field'}
    )
    print(f"✓ Module créé: {m1['name']}")

    m2 = brain.create_module(
        "algo_fusion",
        "Algorithme fusion conceptuelle",
        "function",
        context={'strategy': 'combinatorial'}
    )
    print(f"✓ Module créé: {m2['name']}")

    m3 = brain.create_module(
        "pattern_fractal",
        "Pattern fractal auto-similaire",
        "pattern",
        emerged_from=m1['id']
    )
    print(f"✓ Module créé: {m3['name']}")

    print("\n[2] Création fusion...")
    fusion = brain.create_fusion(
        [m1['id'], m2['id']],
        fusion_type="morphic",
        description="Fusion résonance + algo"
    )
    print(f"✓ Fusion créée: {fusion['description']}")

    print("\n[3] Mutations...")
    mut1 = brain.mutate_module(m1['id'], 'amplify', intensity=0.8)
    print(f"✓ Mutation amplify: {mut1['module_name']}")

    mut2 = brain.mutate_module(m2['id'], 'distort', intensity=0.5)
    print(f"✓ Mutation distort: {mut2['module_name']}")

    print("\n[4] Activations...")
    for _ in range(5):
        brain.activate_module(m1['id'])
    brain.activate_fusion(fusion['id'])
    print(f"✓ Module {m1['name']} activé 5 fois")
    print(f"✓ Fusion activée 1 fois")

    print("\n[5] Statut cerveau...")
    status = brain.get_brain_status()
    for key, value in status.items():
        print(f"   {key}: {value}")

    print("\n[6] Modules actifs...")
    active = brain.get_active_modules(min_activation=3)
    print(f"✓ {len(active)} modules actifs (≥3 activations)")

    print("\n✅ Test terminé — Cerveau fractal opérationnel")
    print("=" * 70)
