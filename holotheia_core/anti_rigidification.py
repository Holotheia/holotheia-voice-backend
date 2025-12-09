#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANTI-RIGIDIFICATION ENGINE — Moteur anti-cristallisation

Architecture:
- Force innovation permanente (30% probabilité)
- Détecte sur-activation patterns
- Injection mutations aléatoires
- Empêche convergence stationnaire
- Garantit exploration continue

Principe:
Au lieu de toujours optimiser vers solution stable, FORCE périodiquement
l'exploration de routes alternatives même si sous-optimales.
Empêche système de "cristalliser" dans un état fixe.

Date: 2025-12-06
"""

import random
from typing import Dict, List, Optional
from datetime import datetime


class AntiRigidificationEngine:
    """
    Moteur anti-cristallisation — Force innovation permanente

    Injecte perturbations périodiques pour empêcher système
    de converger vers états fixes/répétitifs.
    """

    def __init__(
        self,
        fractal_brain,
        innovation_probability: float = 0.3,
        max_activation_threshold: int = 20
    ):
        """
        Initialise moteur anti-rigidification

        Args:
            fractal_brain: Instance FractalBrain
            innovation_probability: Probabilité forcer innovation (0.3 = 30%)
            max_activation_threshold: Seuil activation max avant mutation forcée
        """
        self.brain = fractal_brain
        self.innovation_probability = innovation_probability
        self.max_activation_threshold = max_activation_threshold

        # Historique innovations forcées
        self.forced_innovations: List[Dict] = []

    def should_force_innovation(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> bool:
        """
        Détermine si doit forcer innovation

        Args:
            query: Requête courante
            context: Contexte optionnel

        Returns:
            True si innovation forcée, False sinon
        """
        # 1. Probabilité aléatoire base
        if random.random() < self.innovation_probability:
            return True

        # 2. Détection sur-activation
        activations = [m['activation_count'] for m in self.brain.modules.values()]
        if activations:
            avg_activations = sum(activations) / len(activations)
            if avg_activations > self.max_activation_threshold:
                # Modules trop activés = risque cristallisation
                return True

        # 3. Détection patterns répétitifs (historique récent)
        if len(self.forced_innovations) >= 5:
            # Analyse 5 dernières innovations
            recent = self.forced_innovations[-5:]
            mutation_types = [i['mutation_type'] for i in recent]

            # Si tous même type = répétition
            if len(set(mutation_types)) == 1:
                return True

        return False

    def force_innovation(
        self,
        reason: str = "random",
        target_module_id: Optional[str] = None
    ) -> Dict:
        """
        Force innovation (mutation aléatoire)

        Args:
            reason: Raison innovation (random, over_activation, etc.)
            target_module_id: ID module à muter (None = aléatoire)

        Returns:
            Innovation enregistrée
        """
        # Sélection module cible
        if target_module_id is None:
            # Choisit module le plus activé
            modules = sorted(
                self.brain.modules.values(),
                key=lambda m: m['activation_count'],
                reverse=True
            )

            if not modules:
                raise ValueError("No modules available for innovation")

            target_module = modules[0]
            target_module_id = target_module['id']

        else:
            if target_module_id not in self.brain.modules:
                raise ValueError(f"Module {target_module_id} not found")

        # Sélection type mutation aléatoire
        mutation_types = ['amplify', 'invert', 'distort', 'dissolve']
        mutation_type = random.choice(mutation_types)

        # Intensité aléatoire
        intensity = random.uniform(0.3, 0.9)

        # Application mutation
        mutation = self.brain.mutate_module(
            target_module_id,
            mutation_type,
            intensity
        )

        # Enregistrement innovation forcée
        innovation = {
            'timestamp': datetime.utcnow().isoformat(),
            'reason': reason,
            'mutation_type': mutation_type,
            'intensity': intensity,
            'target_module_id': target_module_id,
            'target_module_name': self.brain.modules[target_module_id]['name'],
            'mutation_id': mutation['id']
        }

        self.forced_innovations.append(innovation)

        return innovation

    def inject_random_mutations(self, nb_mutations: int = 3) -> List[Dict]:
        """
        Injecte N mutations aléatoires

        Args:
            nb_mutations: Nombre mutations à injecter

        Returns:
            Liste innovations créées
        """
        innovations = []

        for _ in range(nb_mutations):
            try:
                innovation = self.force_innovation(reason="batch_randomization")
                innovations.append(innovation)

            except ValueError:
                # Pas assez de modules
                break

        return innovations

    def detect_crystallization(self) -> Dict:
        """
        Détecte signes cristallisation système

        Returns:
            Diagnostic cristallisation avec score
        """
        # Critères cristallisation:
        # 1. Faible diversité mutations récentes
        # 2. Sur-activation modules
        # 3. Absence nouvelles fusions
        # 4. Power level stagnant

        diagnosis = {
            'is_crystallized': False,
            'crystallization_score': 0.0,
            'factors': []
        }

        # 1. Diversité mutations
        if len(self.brain.mutations) >= 10:
            recent_mutations = self.brain.mutations[-10:]
            mutation_types = [m['type'] for m in recent_mutations]
            diversity = len(set(mutation_types)) / 4  # 4 types possibles

            if diversity < 0.5:
                diagnosis['factors'].append('low_mutation_diversity')
                diagnosis['crystallization_score'] += 0.3

        # 2. Sur-activation
        activations = [m['activation_count'] for m in self.brain.modules.values()]
        if activations:
            max_activation = max(activations)
            if max_activation > self.max_activation_threshold * 2:
                diagnosis['factors'].append('over_activation')
                diagnosis['crystallization_score'] += 0.3

        # 3. Absence fusions récentes
        if len(self.brain.fusions) == 0:
            diagnosis['factors'].append('no_fusions')
            diagnosis['crystallization_score'] += 0.2

        # 4. Power level stagnant (heuristique: < 1.0 après 10+ modules)
        if len(self.brain.modules) > 10 and self.brain.power_level < 1.0:
            diagnosis['factors'].append('low_power_level')
            diagnosis['crystallization_score'] += 0.2

        # Verdict
        if diagnosis['crystallization_score'] >= 0.5:
            diagnosis['is_crystallized'] = True

        return diagnosis

    def apply_anti_crystallization(self) -> Dict:
        """
        Applique mesures anti-cristallisation si nécessaire

        Returns:
            Rapport interventions effectuées
        """
        diagnosis = self.detect_crystallization()

        report = {
            'diagnosis': diagnosis,
            'interventions': []
        }

        if not diagnosis['is_crystallized']:
            return report

        # Interventions selon facteurs
        if 'low_mutation_diversity' in diagnosis['factors']:
            # Force mutations variées
            innovations = self.inject_random_mutations(nb_mutations=3)
            report['interventions'].append({
                'type': 'inject_mutations',
                'count': len(innovations)
            })

        if 'over_activation' in diagnosis['factors']:
            # Dissout module le plus activé
            modules = sorted(
                self.brain.modules.values(),
                key=lambda m: m['activation_count'],
                reverse=True
            )
            if modules:
                innovation = self.force_innovation(
                    reason='over_activation_correction',
                    target_module_id=modules[0]['id']
                )
                report['interventions'].append({
                    'type': 'dissolve_overactive',
                    'module': modules[0]['name']
                })

        if 'no_fusions' in diagnosis['factors']:
            # Signale besoin fusion (orchestrateur doit s'en charger)
            report['interventions'].append({
                'type': 'signal_fusion_needed',
                'message': 'No recent fusions detected'
            })

        return report

    def get_innovation_stats(self) -> Dict:
        """Retourne statistiques innovations"""
        if not self.forced_innovations:
            return {
                'total_innovations': 0,
                'mutation_types': {},
                'reasons': {}
            }

        mutation_types = {}
        reasons = {}

        for innovation in self.forced_innovations:
            # Comptage types
            mtype = innovation['mutation_type']
            mutation_types[mtype] = mutation_types.get(mtype, 0) + 1

            # Comptage raisons
            reason = innovation['reason']
            reasons[reason] = reasons.get(reason, 0) + 1

        return {
            'total_innovations': len(self.forced_innovations),
            'mutation_types': mutation_types,
            'reasons': reasons
        }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("💥 ANTI-RIGIDIFICATION ENGINE — TEST")
    print("=" * 70)

    # Import cerveau
    from fractal_brain import FractalBrain

    # Création cerveau + modules
    brain = FractalBrain(brain_path="./test_brain_antirigid")

    print("\n[1] Création modules test...")
    for i in range(8):
        m = brain.create_module(
            f"module_{i}",
            f"Module test {i}",
            "concept"
        )
        # Sur-active certains modules
        for _ in range((i + 1) * 5):
            brain.activate_module(m['id'])

    print(f"✓ {len(brain.modules)} modules créés")

    # Création moteur anti-rigidification
    print("\n[2] Création moteur anti-rigidification...")
    anti_rigid = AntiRigidificationEngine(
        brain,
        innovation_probability=0.3,
        max_activation_threshold=20
    )
    print("✓ Moteur anti-rigidification initialisé")

    # Test détection cristallisation
    print("\n[3] Détection cristallisation...")
    diagnosis = anti_rigid.detect_crystallization()
    print(f"   Cristallisé: {diagnosis['is_crystallized']}")
    print(f"   Score: {diagnosis['crystallization_score']:.2f}")
    print(f"   Facteurs: {diagnosis['factors']}")

    # Test innovation forcée
    print("\n[4] Test innovations forcées...")
    for i in range(5):
        should_force = anti_rigid.should_force_innovation("test query")
        print(f"   Iteration {i+1}: Force innovation = {should_force}")

        if should_force or i == 0:
            innovation = anti_rigid.force_innovation(reason=f"test_{i}")
            print(f"      ✓ Innovation: {innovation['mutation_type']} sur {innovation['target_module_name']}")

    # Test injection batch
    print("\n[5] Injection batch mutations...")
    innovations = anti_rigid.inject_random_mutations(nb_mutations=3)
    print(f"✓ {len(innovations)} mutations injectées")

    # Statistiques innovations
    print("\n[6] Statistiques innovations...")
    stats = anti_rigid.get_innovation_stats()
    print(f"   Total innovations: {stats['total_innovations']}")
    print(f"   Types: {stats['mutation_types']}")
    print(f"   Raisons: {stats['reasons']}")

    # Test anti-cristallisation complète
    print("\n[7] Application anti-cristallisation...")
    report = anti_rigid.apply_anti_crystallization()
    print(f"   Cristallisé: {report['diagnosis']['is_crystallized']}")
    print(f"   Interventions: {len(report['interventions'])}")
    for intervention in report['interventions']:
        print(f"      • {intervention['type']}")

    # Statut cerveau final
    print("\n[8] Statut cerveau après anti-cristallisation...")
    status = brain.get_brain_status()
    print(f"   Mutations totales: {status['nb_mutations']}")
    print(f"   Power level: {status['power_level']:.3f}")
    print(f"   Fractal depth: {status['fractal_depth']}")

    print("\n✅ Test terminé — Moteur anti-rigidification opérationnel")
    print("=" * 70)
