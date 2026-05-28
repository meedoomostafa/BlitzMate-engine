"""Pydantic request/response models for the BlitzMate API."""

from typing import Optional
from pydantic import BaseModel, Field
from server.app.config import DEFAULT_DEPTH


class BestMoveRequest(BaseModel):
    fen: str = Field(..., description="Board position in FEN notation.")
    depth: int = Field(
        default=DEFAULT_DEPTH,
        ge=1,
        le=6,
        description="Search depth (1–6).",
    )


class PlayRequest(BaseModel):
    fen: str = Field(..., description="Current board position in FEN notation.")
    user_move: str = Field(..., description="User's move in UCI format (e.g. 'e2e4').")
    depth: int = Field(
        default=DEFAULT_DEPTH,
        ge=1,
        le=6,
        description="Engine search depth for the response move.",
    )


class BestMoveResponse(BaseModel):
    move: Optional[str] = Field(
        None, description="Best move in UCI format, or null if game is over."
    )
    san: Optional[str] = Field(
        None, description="Best move in SAN notation, or null if game is over."
    )
    depth: int = Field(..., description="Search depth used.")
    score_cp: int = Field(
        ..., description="Evaluation in centipawns (from White's perspective)."
    )
    nodes: int = Field(..., description="Nodes searched.")
    time_ms: int = Field(..., description="Search time in milliseconds.")
    game_over: bool = Field(..., description="Whether the game is over.")
    status: str = Field(
        ...,
        description="Game status: checkmate, stalemate, draw, white_to_move, black_to_move, check.",
    )


class PlayResponse(BaseModel):
    fen: str = Field(
        ...,
        description="Board FEN after both moves (or after user move if game ended).",
    )
    engine_move: Optional[str] = Field(
        None, description="Engine's response in UCI format, or null."
    )
    engine_san: Optional[str] = Field(
        None, description="Engine's response in SAN notation, or null."
    )
    legal: bool = Field(..., description="Whether the user's move was legal.")
    game_over: bool = Field(..., description="Whether the game has ended.")
    status: str = Field(
        ...,
        description=(
            "Game status: white_to_move, black_to_move, check, "
            "checkmate, stalemate, draw."
        ),
    )
    error: Optional[str] = Field(
        None, description="Error message if the move was illegal."
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    engine: str = "BlitzMate"
    max_depth: int
