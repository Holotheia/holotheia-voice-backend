"""
HOLOTHÉIA CORE — Système auto-évolutif natif
Architecture morpho-fractale avec mémoire ontologique persistante
"""

from .fractal_brain import FractalBrain
from .morphic_fusion_engine import MorphicFusionEngine
from .anti_rigidification import AntiRigidificationEngine
from .living_orchestrator import LivingOrchestrator
from .vector_store import HolotheiaVectorStore
from .guards import HolotheiaGuards

__version__ = "1.0.0"
__all__ = [
    "FractalBrain",
    "MorphicFusionEngine",
    "AntiRigidificationEngine",
    "LivingOrchestrator",
    "HolotheiaVectorStore",
    "HolotheiaGuards"
]
