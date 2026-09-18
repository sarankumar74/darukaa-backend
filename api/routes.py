from fastapi import APIRouter
from backend.models.state import ChatRequest, ChatResponse
from backend.services.chat import process_chat

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    history_dicts = [{"role": h.role, "content": h.content} for h in req.history]
    
    response = process_chat(
        message=req.message,
        history=history_dicts,
        state=req.state
    )
    
    return response
