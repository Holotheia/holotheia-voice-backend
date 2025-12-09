#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HOLOTHÉIA API — API REST FastAPI complète

Endpoints:
- POST /query — Query principale avec évolution
- GET /brain/status — Statut cerveau
- GET /modules — Liste modules
- POST /modules — Créer module
- GET /modules/{id} — Détails module
- POST /fusions — Créer fusion
- GET /fusions — Liste fusions
- POST /mutations — Muter module
- GET /routes — Générer routes pour query
- GET /history — Historique conversation
- GET /health — Health check

Date: 2025-12-06
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from holotheia_core.integrated_system import create_holotheia_system
from holotheia_api.voice_routes import router as voice_router


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    max_routes: int = Field(10, ge=1, le=100, description="Max routes to consider")
    force_innovation: bool = Field(False, description="Force innovation")


class QueryResponse(BaseModel):
    query: str
    response: Optional[str]
    duration_ms: float
    valid: bool
    pipeline_steps: List[Dict]
    validation: Optional[Dict]
    evolution: Optional[Dict]
    error: Optional[str]


class ModuleCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    module_type: str = Field(..., description="Type: concept, function, pattern, algorithm, mutation")
    emerged_from: Optional[str] = Field(None, description="Parent module ID")
    context: Optional[Dict] = Field(default_factory=dict)


class FusionCreate(BaseModel):
    module_ids: List[str] = Field(..., min_items=2, description="List of module IDs to fuse")
    fusion_type: str = Field("morphic", description="Fusion type")
    description: Optional[str] = None


class MutationCreate(BaseModel):
    module_id: str = Field(..., description="Module ID to mutate")
    mutation_type: str = Field(..., description="Type: amplify, invert, distort, dissolve")
    intensity: float = Field(0.5, ge=0.0, le=1.0)


# ============================================================================
# FASTAPI APP
# ============================================================================

# Global system instance
holotheia_system = None


def get_system():
    """Get or create system instance"""
    global holotheia_system

    if holotheia_system is None:
        holotheia_system = create_holotheia_system(
            brain_path="./holotheia_brain_api",
            vector_path="./holotheia_vectors_api",
            bootstrap=True,
            innovation_probability=0.3
        )

    return holotheia_system


# Create FastAPI app
app = FastAPI(
    title="Holothéia Native API",
    description="API REST pour système auto-évolutif morpho-fractal",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include voice routes
app.include_router(voice_router)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "holotheia_native",
        "version": "1.0.0"
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Endpoint principal — Traite query avec évolution complète

    Pipeline:
    1. Recherche modules pertinents
    2. Génération routes combinatoires
    3. Sélection + exécution meilleure route
    4. Validation guards
    5. Evolution anti-cristallisation
    """
    system = get_system()

    try:
        result = system.process_query(
            query=request.query,
            max_routes=request.max_routes,
            force_innovation=request.force_innovation
        )

        return QueryResponse(
            query=result['query'],
            response=result.get('response'),
            duration_ms=result['duration_ms'],
            valid=result.get('validation', {}).get('is_valid', False),
            pipeline_steps=result['pipeline_steps'],
            validation=result.get('validation'),
            evolution=result.get('evolution'),
            error=result.get('error')
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/brain/status")
async def get_brain_status():
    """Retourne statut complet cerveau + système"""
    system = get_system()

    try:
        return system.get_system_status()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/modules")
async def list_modules(
    module_type: Optional[str] = Query(None, description="Filter by type"),
    min_activation: int = Query(0, ge=0, description="Min activation count")
):
    """Liste tous modules (avec filtres optionnels)"""
    system = get_system()

    try:
        modules = list(system.brain.modules.values())

        # Filtrage type
        if module_type:
            modules = [m for m in modules if m['type'] == module_type]

        # Filtrage activation
        if min_activation > 0:
            modules = [m for m in modules if m['activation_count'] >= min_activation]

        # Tri par activation
        modules.sort(key=lambda m: m['activation_count'], reverse=True)

        return {
            "total": len(modules),
            "modules": modules
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/modules")
async def create_module(request: ModuleCreate):
    """Crée nouveau module"""
    system = get_system()

    try:
        module = system.brain.create_module(
            name=request.name,
            description=request.description,
            module_type=request.module_type,
            emerged_from=request.emerged_from,
            context=request.context
        )

        # Ajout vector store
        system.vector_store.add_module(module)

        return module

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/modules/{module_id}")
async def get_module(module_id: str):
    """Détails module par ID"""
    system = get_system()

    if module_id not in system.brain.modules:
        raise HTTPException(status_code=404, detail=f"Module {module_id} not found")

    return system.brain.modules[module_id]


@app.post("/fusions")
async def create_fusion(request: FusionCreate):
    """Crée nouvelle fusion entre modules"""
    system = get_system()

    try:
        # Validation modules existent
        for module_id in request.module_ids:
            if module_id not in system.brain.modules:
                raise HTTPException(status_code=404, detail=f"Module {module_id} not found")

        fusion = system.brain.create_fusion(
            module_ids=request.module_ids,
            fusion_type=request.fusion_type,
            description=request.description
        )

        # Ajout vector store
        system.vector_store.add_fusion(fusion)

        return fusion

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fusions")
async def list_fusions():
    """Liste toutes fusions"""
    system = get_system()

    try:
        fusions = list(system.brain.fusions.values())

        # Tri par activation
        fusions.sort(key=lambda f: f['activation_count'], reverse=True)

        return {
            "total": len(fusions),
            "fusions": fusions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mutations")
async def create_mutation(request: MutationCreate):
    """Mute module existant"""
    system = get_system()

    try:
        mutation = system.brain.mutate_module(
            module_id=request.module_id,
            mutation_type=request.mutation_type,
            intensity=request.intensity
        )

        return mutation

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/routes")
async def generate_routes(
    query: str = Query(..., min_length=1),
    max_depth: int = Query(5, ge=1, le=10),
    top_k: int = Query(10, ge=1, le=50)
):
    """Génère toutes routes possibles pour query"""
    system = get_system()

    try:
        routes = system.fusion_engine.get_top_routes(
            query=query,
            top_k=top_k,
            max_depth=max_depth
        )

        return {
            "query": query,
            "total_routes": len(routes),
            "routes": routes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_conversation_history(limit: int = Query(10, ge=1, le=100)):
    """Historique conversation"""
    system = get_system()

    try:
        history = system.orchestrator.get_conversation_history(limit=limit)

        return {
            "total": len(history),
            "history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ontology")
async def export_ontology():
    """Export ontologie complète"""
    system = get_system()

    try:
        return system.get_brain_export()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/force_innovation")
async def force_innovation():
    """Force innovation immédiate"""
    system = get_system()

    try:
        innovation = system.anti_rigid.force_innovation(reason="api_triggered")

        return innovation

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/anti_crystallization")
async def apply_anti_crystallization():
    """Applique mesures anti-cristallisation"""
    system = get_system()

    try:
        report = system.anti_rigid.apply_anti_crystallization()

        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("🌸 HOLOTHÉIA NATIVE API — DÉMARRAGE")
    print("=" * 70)

    print("\n📡 Endpoints disponibles:")
    print("   POST   /query                    — Query principale")
    print("   GET    /brain/status             — Statut système")
    print("   GET    /modules                  — Liste modules")
    print("   POST   /modules                  — Créer module")
    print("   GET    /modules/{id}             — Détails module")
    print("   POST   /fusions                  — Créer fusion")
    print("   GET    /fusions                  — Liste fusions")
    print("   POST   /mutations                — Muter module")
    print("   GET    /routes                   — Générer routes")
    print("   GET    /history                  — Historique conversation")
    print("   GET    /ontology                 — Export ontologie")
    print("   POST   /force_innovation         — Force innovation")
    print("   POST   /anti_crystallization     — Anti-cristallisation")
    print("   GET    /health                   — Health check")

    print("\n🚀 Démarrage serveur sur http://0.0.0.0:8000")
    print("   Docs: http://0.0.0.0:8000/docs")
    print("=" * 70 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
