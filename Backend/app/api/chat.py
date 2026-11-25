"""
app/api/chat.py
───────────────
AI Security Analyst chatbot endpoint.

POST /api/chat – accept a user question, run the LangChain agent,
                 return a grounded answer with source references.

The agent is imported lazily so a missing GROQ_API_KEY does not crash
the entire API on startup — it only fails when the chat endpoint is called.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Ask the AI Security Analyst a question about historical drone footage.

    The agent queries the database through controlled tools before answering,
    so every answer is grounded in actual stored records.
    """
    # Lazy import — keeps startup fast and isolates the LangChain dependency
    try:
        from app.services.agent import answer_question
    except ImportError as exc:
        logger.error("Agent import failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="AI agent is unavailable. Check GROQ_API_KEY and dependencies.",
        )

    try:
        response = answer_question(
            question=request.question,
            session_id=request.session_id,
            db=db,
        )
        return response
    except Exception as exc:
        logger.exception("Chat agent error for session %s: %s", request.session_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {exc}",
        )
