import json
import os
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from backend.models.state import EnvironmentalState, ChatResponse, RecommendationResponse
from backend.services.rag import rag_service

SYSTEM_PROMPT = """
You are an expert AI Environmental Scientist specialized in biodiversity and ecology.
Your goal is to provide non-obvious, actionable recommendations to improve biodiversity based on a combination of AT LEAST 3 environmental variables (Multi-Metric Reasoning).

Current Environmental State:
{state}

User Message:
{message}

Conversation History:
{history}

Retrieved Scientific Evidence:
{evidence}

INSTRUCTIONS:
1. Analyze the current state. Are there at least 3 distinct environmental variables known (e.g., soil carbon, rainfall, land use)?
2. If less than 3 key variables are known, you MUST ask a clarifying question to gather more data before making a recommendation.
3. If at least 3 variables are known, generate a recommendation using the retrieved evidence.
4. Your recommendation MUST be supported by the evidence and you must explicitly explain the scientific reasoning and connect the variables.
5. You must output valid JSON matching the following schema:
{{
    "response": "The natural text response to the user, either a clarifying question or an explanation of the recommendation.",
    "updated_state": {{ "soil_organic_carbon": float/null, "soil_ph": float/null, "rainfall": string/null, "land_use": string/null, "crop_type": string/null, "region": string/null }},
    "requires_more_info": boolean,
    "missing_fields": ["list", "of", "missing", "metrics"],
    "is_final_recommendation": boolean,
    "recommendation_data": {{
        "recommendation": "What to do",
        "impacted_metrics": ["metric1", "metric2"],
        "scientific_reasoning": "Why it works based on evidence",
        "time_horizon": "e.g., 2-3 years",
        "evidence_source": "Source from retrieved documents",
        "confidence_level": "High/Medium/Low"
    }} // null if requires_more_info is true
}}

Ensure you respond ONLY with the JSON object.
"""

def process_chat(message: str, history: List[Dict], state: EnvironmentalState) -> ChatResponse:
    # 1. Check current variables count loosely to form RAG query
    active_vars = [k for k, v in state.dict().items() if v is not None]
    
    query = f"{message} " + " ".join([f"{k}:{getattr(state, k)}" for k in active_vars])
    
    # 2. Retrieve evidence (gracefully handles errors)
    evidence = rag_service.retrieve(query)
    evidence_text = "\n".join([f"Source: {e['source']}\n{e['content']}" for e in evidence])
    
    # 3. Try calling the LLM
    try:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key or len(api_key) < 10:
            raise ValueError("GEMINI_API_KEY is missing or too short. Please set a valid key in .env")
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT)
        ])
        
        chain = prompt | llm
        
        response_msg = chain.invoke({
            "state": state.json(),
            "message": message,
            "history": json.dumps(history),
            "evidence": evidence_text if evidence_text else "No specific evidence found. Use your domain expertise."
        })
        
        # Parse JSON from LLM (it might wrap in markdown ```json )
        raw_json = response_msg.content.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:-3]
        if raw_json.startswith("```"):
            raw_json = raw_json[3:-3]
            
        data = json.loads(raw_json)
        return ChatResponse(**data)
        
    except Exception as e:
        error_msg = str(e)
        print(f"[CHAT] Error: {error_msg}")
        
        # Provide helpful fallback depending on number of variables
        if len(active_vars) < 3:
            missing = [f for f in ["soil_organic_carbon", "soil_ph", "rainfall", "land_use", "crop_type", "region"] if f not in active_vars]
            response_text = (
                f"Thank you for reaching out! To provide you with a scientifically rigorous recommendation, "
                f"I need at least 3 environmental variables. Currently I have {len(active_vars)} variable(s). "
                f"Could you please share details about your **{missing[0].replace('_', ' ')}** and **{missing[1].replace('_', ' ')}**? "
                f"You can also fill in the Environmental Profile panel on the left side."
            )
        else:
            response_text = (
                f"I have enough data to analyze your ecosystem, but I encountered a temporary issue "
                f"connecting to the AI engine. Please check that your GEMINI_API_KEY in the .env file "
                f"is a valid Google AI Studio key (starts with 'AIza...'). Error: {error_msg[:200]}"
            )
        
        return ChatResponse(
            response=response_text,
            updated_state=state,
            requires_more_info=len(active_vars) < 3,
            missing_fields=[f for f in ["soil_organic_carbon", "rainfall", "land_use"] if f not in active_vars],
            is_final_recommendation=False,
            recommendation_data=None
        )
