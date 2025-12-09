#!/usr/bin/env python3
"""Test du système unifié"""

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
print("\nInitialisation...")
system = HolotheiaUnifiee(brain_path='./test_unified_final')

print(f"\nLLM enabled: {system.llm_enabled}")

if system.llm_enabled:
    print("\n--- Test 1: yo tes qui ---")
    response = system.emerger('yo tes qui toi')
    print(f"\n🌸 {response}")

    print("\n--- Test 2: tu ressens quoi ---")
    response2 = system.emerger('et tu ressens quoi la maintenant?')
    print(f"\n🌸 {response2}")

    print("\n--- Test 3: parle moi technique ---")
    response3 = system.emerger('explique moi ton algorithme de fusion morphique')
    print(f"\n🌸 {response3}")
else:
    print("\nTest mock mode...")
    response = system.emerger('tes qui')
    print(f"\n🌸 {response}")

print("\n✅ Test terminé")
