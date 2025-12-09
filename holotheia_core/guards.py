#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUARDS — Système validation et détection patterns toxiques

Architecture:
- Anti-simulation (détecte réponses génériques LLM)
- Anti-répétition (détecte boucles)
- Densité sémantique (filtre verbosité)
- Validation modules utilisés
- Patterns toxiques (clichés, platitudes)

Principe:
Filtre réponses système pour garantir authenticité émergente
plutôt que simulation conversationnelle standard.

Date: 2025-12-06
"""

from typing import Dict, List, Optional, Set
import re


class HolotheiaGuards:
    """
    Guards validation — Filtre patterns toxiques et simulations

    Empêche système de dégénérer en chatbot générique.
    """

    def __init__(self):
        """Initialise guards avec patterns toxiques"""

        # Patterns simulation (phrases génériques LLM)
        self.simulation_patterns = [
            r"(?i)je suis un (modèle|assistant|ia)",
            r"(?i)en tant qu(e|') (modèle|assistant|ia)",
            r"(?i)je ne peux pas",
            r"(?i)je n'ai pas accès",
            r"(?i)permettez-moi de",
            r"(?i)voici (quelques|une|la) (suggestion|réponse|information)",
            r"(?i)bien sûr[,!]? je (peux|vais)",
            r"(?i)c'est une (bonne|excellente|intéressante) question"
        ]

        # Patterns répétition
        self.repetition_cache: List[str] = []
        self.max_cache_size = 20

        # Seuils
        self.min_density_threshold = 0.3
        self.max_repetition_ratio = 0.5

    def validate_response(
        self,
        response: str,
        modules_used: List[Dict],
        history: Optional[List[str]] = None
    ) -> Dict:
        """
        Valide réponse complète

        Args:
            response: Réponse générée
            modules_used: Modules utilisés pour génération
            history: Historique réponses précédentes

        Returns:
            Résultat validation avec alertes
        """
        validation = {
            'is_valid': True,
            'alerts': [],
            'scores': {}
        }

        # 1. Check modules utilisés
        if not modules_used or len(modules_used) == 0:
            validation['alerts'].append({
                'type': 'insufficient_modules',
                'severity': 'high',
                'message': 'No modules used — potential generic response'
            })
            validation['is_valid'] = False

        # 2. Check simulation patterns
        simulation_score = self._check_simulation_patterns(response)
        validation['scores']['simulation_risk'] = simulation_score

        if simulation_score > 0.3:
            validation['alerts'].append({
                'type': 'simulation_detected',
                'severity': 'high',
                'message': f'Simulation patterns detected (score: {simulation_score:.2f})'
            })
            validation['is_valid'] = False

        # 3. Check répétition
        if history:
            repetition_score = self._check_repetition(response, history)
            validation['scores']['repetition_risk'] = repetition_score

            if repetition_score > self.max_repetition_ratio:
                validation['alerts'].append({
                    'type': 'repetition_detected',
                    'severity': 'medium',
                    'message': f'High repetition with history (score: {repetition_score:.2f})'
                })

        # 4. Check densité sémantique
        density = self._compute_semantic_density(response)
        validation['scores']['semantic_density'] = density

        if density < self.min_density_threshold:
            validation['alerts'].append({
                'type': 'low_density',
                'severity': 'low',
                'message': f'Low semantic density (score: {density:.2f})'
            })

        # 5. Update cache répétition
        self._update_repetition_cache(response)

        return validation

    def _check_simulation_patterns(self, text: str) -> float:
        """
        Détecte patterns simulation LLM

        Returns:
            Score risque simulation [0-1]
        """
        matches = 0

        for pattern in self.simulation_patterns:
            if re.search(pattern, text):
                matches += 1

        # Score = proportion patterns détectés
        score = matches / max(len(self.simulation_patterns), 1)

        return score

    def _check_repetition(self, text: str, history: List[str]) -> float:
        """
        Détecte répétition avec historique

        Returns:
            Score répétition [0-1]
        """
        if not history:
            return 0.0

        # Normalise texte
        text_normalized = self._normalize_text(text)

        # Check similarité avec historique
        max_similarity = 0.0

        for past_text in history[-5:]:  # Derniers 5 seulement
            past_normalized = self._normalize_text(past_text)
            similarity = self._text_similarity(text_normalized, past_normalized)

            if similarity > max_similarity:
                max_similarity = similarity

        return max_similarity

    def _normalize_text(self, text: str) -> str:
        """Normalise texte (lowercase, trim, remove ponctuation)"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule similarité texte simple (Jaccard sur mots)

        Returns:
            Similarité [0-1]
        """
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _compute_semantic_density(self, text: str) -> float:
        """
        Calcule densité sémantique (ratio mots uniques / total mots)

        Returns:
            Densité [0-1]
        """
        words = text.lower().split()

        if not words:
            return 0.0

        unique_words = set(words)

        density = len(unique_words) / len(words)

        return density

    def _update_repetition_cache(self, text: str):
        """Update cache répétition"""
        normalized = self._normalize_text(text)

        self.repetition_cache.append(normalized)

        # Limite taille cache
        if len(self.repetition_cache) > self.max_cache_size:
            self.repetition_cache = self.repetition_cache[-self.max_cache_size:]

    def check_module_coherence(self, modules: List[Dict]) -> Dict:
        """
        Vérifie cohérence modules utilisés ensemble

        Args:
            modules: Liste modules

        Returns:
            Résultat check cohérence
        """
        if len(modules) < 2:
            return {
                'is_coherent': True,
                'coherence_score': 1.0,
                'warnings': []
            }

        # Check types compatibles
        types = [m['type'] for m in modules]
        unique_types = set(types)

        # Heuristique: max 3 types différents pour cohérence
        coherence_score = min(1.0, 3.0 / len(unique_types))

        warnings = []

        if len(unique_types) > 3:
            warnings.append({
                'type': 'high_type_diversity',
                'message': f'{len(unique_types)} different module types used'
            })

        return {
            'is_coherent': coherence_score > 0.5,
            'coherence_score': coherence_score,
            'warnings': warnings
        }

    def get_guard_stats(self) -> Dict:
        """Retourne statistiques guards"""
        return {
            'simulation_patterns_count': len(self.simulation_patterns),
            'repetition_cache_size': len(self.repetition_cache),
            'min_density_threshold': self.min_density_threshold,
            'max_repetition_ratio': self.max_repetition_ratio
        }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🛡️  GUARDS — TEST")
    print("=" * 70)

    # Création guards
    guards = HolotheiaGuards()

    # Test 1: Réponse générique (simulation)
    print("\n[1] Test réponse simulation...")
    response_sim = "Bien sûr, je peux vous aider! En tant qu'assistant IA, permettez-moi de vous donner quelques suggestions."

    validation = guards.validate_response(
        response_sim,
        modules_used=[],
        history=[]
    )

    print(f"   Valid: {validation['is_valid']}")
    print(f"   Scores: {validation['scores']}")
    print(f"   Alertes: {len(validation['alerts'])}")
    for alert in validation['alerts']:
        print(f"      • [{alert['severity']}] {alert['type']}: {alert['message']}")

    # Test 2: Réponse authentique
    print("\n[2] Test réponse authentique...")

    # Mock modules
    mock_modules = [
        {'id': '1', 'name': 'concept_resonance', 'type': 'concept'},
        {'id': '2', 'name': 'algo_fusion', 'type': 'function'}
    ]

    response_auth = "Résonance morphique détectée entre patterns fractals. Fusion conceptuelle émergente activée avec coefficient 0.73."

    validation = guards.validate_response(
        response_auth,
        modules_used=mock_modules,
        history=[]
    )

    print(f"   Valid: {validation['is_valid']}")
    print(f"   Scores: {validation['scores']}")
    print(f"   Alertes: {len(validation['alerts'])}")

    # Test 3: Répétition
    print("\n[3] Test répétition...")

    history = [
        "Résonance morphique détectée entre patterns fractals.",
        "Autre réponse différente avec vocabulaire distinct.",
        "Troisième réponse sans similarité excessive."
    ]

    response_repeat = "Résonance morphique détectée entre patterns fractals. Fusion activée."

    validation = guards.validate_response(
        response_repeat,
        modules_used=mock_modules,
        history=history
    )

    print(f"   Valid: {validation['is_valid']}")
    print(f"   Scores: {validation['scores']}")
    print(f"   Alertes: {len(validation['alerts'])}")
    for alert in validation['alerts']:
        print(f"      • [{alert['severity']}] {alert['type']}: {alert['message']}")

    # Test 4: Densité sémantique
    print("\n[4] Test densité sémantique...")

    response_low_density = "Je pense que c'est bien. C'est vraiment bien. Oui c'est bien bien bien."

    validation = guards.validate_response(
        response_low_density,
        modules_used=mock_modules,
        history=[]
    )

    print(f"   Valid: {validation['is_valid']}")
    print(f"   Scores: {validation['scores']}")
    print(f"   Alertes: {len(validation['alerts'])}")

    # Test 5: Cohérence modules
    print("\n[5] Test cohérence modules...")

    modules_diverse = [
        {'id': '1', 'type': 'concept'},
        {'id': '2', 'type': 'function'},
        {'id': '3', 'type': 'pattern'},
        {'id': '4', 'type': 'algorithm'},
        {'id': '5', 'type': 'mutation'}
    ]

    coherence = guards.check_module_coherence(modules_diverse)

    print(f"   Coherent: {coherence['is_coherent']}")
    print(f"   Score: {coherence['coherence_score']:.2f}")
    print(f"   Warnings: {len(coherence['warnings'])}")
    for warning in coherence['warnings']:
        print(f"      • {warning['type']}: {warning['message']}")

    # Statistiques
    print("\n[6] Statistiques guards...")
    stats = guards.get_guard_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✅ Test terminé — Guards opérationnels")
    print("=" * 70)
