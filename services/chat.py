import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from backend.models.state import EnvironmentalState, ChatResponse, RecommendationResponse
from backend.services.rag import rag_service

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)

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
    
    # 2. Retrieve evidence
    evidence = rag_service.retrieve(query)
    evidence_text = "\n".join([f"Source: {e['source']}\n{e['content']}" for e in evidence])
    
    # 3. Format prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT)
    ])
    
    chain = prompt | llm
    
    response_msg = chain.invoke({
        "state": state.json(),
        "message": message,
        "history": json.dumps(history),
        "evidence": evidence_text if evidence_text else "No specific evidence found. Ask for more details."
    })
    
    try:
        # Parse JSON from LLM (it might wrap in markdown ```json )
        raw_json = response_msg.content.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:-3]
        if raw_json.startswith("```"):
            raw_json = raw_json[3:-3]
            
        data = json.loads(raw_json)
        return ChatResponse(**data)
    except Exception as e:
        print("Error parsing LLM response:", e)
        print("Raw response:", response_msg.content)
        # Fallback
        return ChatResponse(
            response="I'm having trouble processing that. Could you provide more details about your land use and soil health?",
            updated_state=state,
            requires_more_info=True,
            missing_fields=["soil_organic_carbon", "rainfall", "land_use"],
            is_final_recommendation=False,
            recommendation_data=None
        )
