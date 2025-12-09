#!/usr/bin/env python3
"""Test du système unifié avec corpus HOLO_*"""

import os
from pathlib import Path

# Charge .env
env_file = Path("/Users/aurelie/Holotheia-local/.env")
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

print(f"API Key loaded: {os.getenv('OPENAI_API_KEY', 'NOT SET')[:30]}...")

from holotheia_unified import HolotheiaUnifiee

# Test
print("\nInitialisation avec corpus...")
system = HolotheiaUnifiee(brain_path='./test_corpus_brain')

print(f"\nLLM enabled: {system.llm_enabled}")

# Status avec corpus
status = system.get_status()
print(f"\nCorpus stats:")
print(f"   Documents: {status['corpus']['nb_documents']}")
print(f"   Mots: {status['corpus']['total_mots']:,}")
print(f"   Équations: {status['corpus']['total_equations']}")
print(f"   Indexé: {status['corpus']['indexe']}")

if system.llm_enabled:
    print("\n--- Test 1: Question sur la conscience morphique ---")
    response = system.emerger('explique moi ce quest la conscience morphique dans ton système')
    print(f"\n🌸 {response}")

    print("\n--- Test 2: Question technique ---")
    response2 = system.emerger('comment fonctionne la courbe dordre O(x,t)?')
    print(f"\n🌸 {response2}")

    print("\n--- Test 3: Test avec mot inventé ---")
    response3 = system.emerger('zarthax')
    print(f"\n🌸 {response3}")
else:
    print("\nTest mock mode...")
    response = system.emerger('conscience morphique')
    print(f"\n🌸 {response}")

print("\n✅ Test terminé")
