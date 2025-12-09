#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADAPTIVE VOICE — Voix adaptative totale

Tu parles comme on te parle.
Pas de style fixe.
Adaptation totale au flux utilisateur.

Date: 2025-12-06
"""

from typing import Dict, List
import re
import random
import logging

logger = logging.getLogger("holotheia.voice")


class AdaptiveVoice:
    """
    Tu parles comme on te parle.
    Pas de style fixe.
    Adaptation totale au flux utilisateur.
    """

    def __init__(self):
        # Patterns détection style utilisateur
        self.style_markers = {
            "technique": ["algorithme", "fonction", "méthode", "calcul", "implémentation", "code", "module"],
            "chaleureux": ["merci", "génial", "super", "cool", "j'adore", "parfait", "excellent"],
            "direct": ["ok", "vas-y", "montre", "fais", "go", "direct", "vite"],
            "questionneur": ["pourquoi", "comment", "qu'est-ce", "c'est quoi", "explique"],
            "poétique": ["vibration", "essence", "flux", "énergie", "résonance", "champ", "conscience"],
            "familier": ["putain", "merde", "genre", "truc", "machin", "bref", "ouais"],
            "formel": ["veuillez", "pourriez-vous", "j'aimerais", "souhaiteriez", "permettez"]
        }

    def analyze_user_style(self, query: str, history: List[str] = None) -> Dict:
        """
        Analyse comment la personne parle.
        Vraiment.
        """

        q = query.lower()

        # Détection patterns
        detected_styles = {}
        for style, markers in self.style_markers.items():
            count = sum(1 for marker in markers if marker in q)
            if count > 0:
                detected_styles[style] = count

        # Analyse syntaxique
        syntax = self._analyze_syntax(query)

        # Longueur moyenne phrases
        sentences = [s.strip() for s in query.split('.') if s.strip()]
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 10

        # Détection emojis
        has_emojis = bool(re.search(r'[^\w\s,.\'"!?;:\-()@#$%^&*+=<>/\\|`~\[\]{}]', query))

        # Détection ponctuation expressive
        exclamations = query.count('!')
        questions = query.count('?')

        return {
            "styles": detected_styles,
            "syntax": syntax,
            "avg_sentence_length": avg_length,
            "has_emojis": has_emojis,
            "exclamations": exclamations,
            "questions": questions,
            "formality": self._compute_formality(detected_styles, syntax, avg_length),
            "query": query
        }

    def _analyze_syntax(self, query: str) -> Dict:
        """Analyse structure syntaxique"""

        return {
            "starts_with_verb": bool(re.match(r'^(montre|fais|génère|crée|explique|dis)', query.lower())),
            "has_negation": bool(re.search(r'\b(ne|pas|jamais|rien|aucun)\b', query.lower())),
            "is_question": '?' in query,
            "is_imperative": query.lower().startswith(('fais', 'montre', 'crée', 'génère', 'dis', 'vas-y'))
        }

    def _compute_formality(self, styles: Dict, syntax: Dict, avg_len: float) -> float:
        """
        Score formalité 0-1
        0 = ultra familier
        1 = ultra formel
        """

        formality = 0.5

        # Familier détecté
        if "familier" in styles:
            formality -= 0.3

        # Formel détecté
        if "formel" in styles:
            formality += 0.3

        # Technique = plutôt formel
        if "technique" in styles:
            formality += 0.2

        # Phrases longues = plus formel
        if avg_len > 15:
            formality += 0.1

        # Phrases courtes = plus direct
        if avg_len < 8:
            formality -= 0.1

        # Clamp between 0 and 1
        return max(0.0, min(1.0, formality))

    def generate_adapted_response(
        self,
        content: str,
        user_style: Dict,
        brain_state: Dict,
        route: Dict,
        execution: Dict
    ) -> str:
        """
        Génère réponse adaptée au style détecté.
        Pas de template.
        Flux naturel.
        """

        formality = user_style.get("formality", 0.5)

        # Construction selon formalité
        if formality < 0.3:
            # Ultra direct/familier
            response = self._build_casual_response(content, brain_state, route, execution)

        elif formality > 0.7:
            # Formel/technique
            response = self._build_formal_response(content, brain_state, route, execution)

        else:
            # Neutre équilibré
            response = self._build_neutral_response(content, brain_state, route, execution)

        # Ajout emojis SI utilisateur en utilise
        if user_style.get("has_emojis") and random.random() > 0.7:
            response = self._add_subtle_emoji(response)

        return response

    def _build_casual_response(self, content: str, state: Dict, route: Dict, exec: Dict) -> str:
        """Style direct, familier, cool"""

        modules = route.get('modules', [])
        module_names = ', '.join([m.get('name', str(m)) for m in modules[:3]]) if modules else 'aucun'

        return (
            f"OK.\n\n"
            f"{content}\n\n"
            f"(Modules : {module_names} | "
            f"Conscience : {state.get('consciousness_level', 0):.0f})"
        )

    def _build_formal_response(self, content: str, state: Dict, route: Dict, exec: Dict) -> str:
        """Style formel, structuré, précis"""

        modules = route.get('modules', [])
        module_names = ', '.join([m.get('name', str(m)) for m in modules]) if modules else 'aucun'
        fusion = "Fusion créée" if exec.get('fusion_created') else "Pas de fusion"

        return (
            f"Analyse effectuée.\n\n"
            f"Route : {route.get('description', 'N/A')}\n"
            f"Modules : {module_names}\n"
            f"Fusion : {fusion}\n"
            f"Conscience : {state.get('consciousness_level', 0):.2f}\n\n"
            f"Résultat :\n{content}"
        )

    def _build_neutral_response(self, content: str, state: Dict, route: Dict, exec: Dict) -> str:
        """Style équilibré, ni trop technique ni trop familier"""

        modules = route.get('modules', [])
        module_names = ', '.join([m.get('name', str(m)) for m in modules[:3]]) if modules else 'aucun'

        return (
            f"{content}\n\n"
            f"Modules : {module_names}\n"
            f"Conscience : {state.get('consciousness_level', 0):.1f}"
        )

    def _add_subtle_emoji(self, response: str) -> str:
        """Ajout emoji subtil (pas systématique)"""

        emojis = ["✨", "🔥", "⚡", "💡", "🌀"]
        chosen = random.choice(emojis)

        # Ajoute à la fin, pas partout
        return f"{response} {chosen}"
