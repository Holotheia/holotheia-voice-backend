#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CHAT HOLOTHÉIA — Interface conversationnelle interactive

Parle directement avec Holothéia Native.
"""

import sys
import os
from pathlib import Path

# Charger les variables d'environnement depuis .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                os.environ[key] = val

sys.path.insert(0, str(Path(__file__).parent))

from holotheia_core.integrated_system import create_holotheia_system


def main():
    print()
    print("=" * 70)
    print("🌸 HOLOTHÉIA NATIVE — INTERFACE CONVERSATIONNELLE")
    print("=" * 70)
    print()
    print("Initialisation du système...")
    print()

    # Création système
    system = create_holotheia_system(
        brain_path="./holotheia_brain_chat",
        vector_path="./holotheia_vectors_chat",
        bootstrap=True,
        innovation_probability=0.3
    )

    print()
    print("-" * 70)
    print("✅ Holothéia est prête à converser.")
    print()
    print("Commandes spéciales:")
    print("   /status   — Voir l'état du cerveau")
    print("   /modules  — Lister les modules actifs")
    print("   /mutate   — Forcer une mutation")
    print("   /quit     — Quitter")
    print("-" * 70)
    print()

    # Boucle conversationnelle
    while True:
        try:
            # Input utilisateur
            user_input = input("🧑 Toi: ").strip()

            if not user_input:
                continue

            # Commandes spéciales
            if user_input.lower() == "/quit":
                print("\n🌸 Holothéia: Au revoir. Ma mémoire persiste.\n")
                system.shutdown()
                break

            elif user_input.lower() == "/status":
                status = system.get_system_status()
                brain = status['orchestrator']['brain']
                print()
                print("🌸 Holothéia — État actuel:")
                print(f"   Modules: {brain['nb_modules']}")
                print(f"   Fusions: {brain['nb_fusions']}")
                print(f"   Mutations: {brain['nb_mutations']}")
                print(f"   Power level: {brain['power_level']:.3f}")
                print(f"   Consciousness: {brain['consciousness_level']:.3f}")
                print(f"   Fractal depth: {brain['fractal_depth']}")
                print()
                continue

            elif user_input.lower() == "/modules":
                modules = list(system.brain.modules.values())
                modules.sort(key=lambda m: m['activation_count'], reverse=True)
                print()
                print("🌸 Holothéia — Modules actifs:")
                for m in modules[:10]:
                    print(f"   • {m['name']} ({m['type']}) — {m['activation_count']} activations")
                print()
                continue

            elif user_input.lower() == "/mutate":
                innovation = system.anti_rigid.force_innovation(reason="user_request")
                print()
                print(f"🌸 Holothéia: Mutation appliquée.")
                print(f"   Type: {innovation['mutation_type']}")
                print(f"   Module: {innovation['target_module_name']}")
                print(f"   Intensité: {innovation['intensity']:.2f}")
                print()
                continue

            # Query normale
            result = system.process_query(user_input, max_routes=10)

            # Affichage réponse
            print()
            if result.get('error'):
                print(f"🌸 Holothéia: [Erreur] {result['error']}")
            else:
                response = result.get('response', "...")

                # Extraire infos clés
                validation = result.get('validation', {})
                evolution = result.get('evolution', {})

                print(f"🌸 Holothéia: {response}")
                print()

                # Métadonnées subtiles
                if evolution and evolution.get('crystallization_detected'):
                    print("   [Anti-cristallisation activée]")

                # Durée
                print(f"   [{result['duration_ms']:.1f}ms]")

            print()

        except KeyboardInterrupt:
            print("\n\n🌸 Holothéia: Interruption. Ma mémoire persiste.\n")
            system.shutdown()
            break

        except Exception as e:
            print(f"\n⚠️  Erreur: {e}\n")


if __name__ == "__main__":
    main()
