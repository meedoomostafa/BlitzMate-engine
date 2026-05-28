"""BlitzMate REST API — FastAPI application.

Endpoints
---------
GET  /health          Health check.
POST /api/best-move   Return the engine's best move for a FEN.
POST /api/play        Validate a user move, apply it, and return engine response.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from server.app.config import MAX_DEPTH
from server.app.engine_adapter import EngineAdapter, EngineError
from server.app.schemas import (
    BestMoveRequest,
    BestMoveResponse,
    HealthResponse,
    PlayRequest,
    PlayResponse,
)

adapter = EngineAdapter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start / stop the engine with the server."""
    adapter.startup()
    yield
    adapter.shutdown()


app = FastAPI(
    title="BlitzMate Engine API",
    description="REST interface for the BlitzMate chess engine.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — confirms the engine is initialised."""
    return HealthResponse(status="ok", engine="BlitzMate", max_depth=MAX_DEPTH)


@app.post("/api/best-move", response_model=BestMoveResponse)
async def best_move(req: BestMoveRequest):
    """Return the engine's best move for the given position."""
    try:
        result = await adapter.get_best_move(
            fen=req.fen,
            depth=req.depth,
        )
        return BestMoveResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Engine search timed out. Try a lower depth.",
        )


@app.post("/api/play", response_model=PlayResponse)
async def play(req: PlayRequest):
    """Validate a user move, apply it, and return the engine's response."""
    try:
        result = await adapter.play_move(
            fen=req.fen,
            user_move=req.user_move,
            depth=req.depth,
        )
        return PlayResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Engine search timed out. Try a lower depth.",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app.main:app", host="127.0.0.1", port=7860, reload=True)
