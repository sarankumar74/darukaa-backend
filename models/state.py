from pydantic import BaseModel, Field
from typing import Optional, List

class EnvironmentalState(BaseModel):
    soil_organic_carbon: Optional[float] = Field(None, description="Percentage of soil organic carbon")
    soil_ph: Optional[float] = Field(None, description="Soil pH level")
    rainfall: Optional[str] = Field(None, description="Rainfall pattern (e.g., low, moderate, heavy)")
    land_use: Optional[str] = Field(None, description="Current land use (e.g., monoculture, agroforestry)")
    crop_type: Optional[str] = Field(None, description="Current crop planted")
    region: Optional[str] = Field(None, description="Geographical region or climate zone")
    
class ChatMessage(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    state: Optional[EnvironmentalState] = None

class RecommendationResponse(BaseModel):
    recommendation: str
    impacted_metrics: List[str]
    scientific_reasoning: str
    time_horizon: str
    evidence_source: str
    confidence_level: str

class ChatResponse(BaseModel):
    response: str
    updated_state: EnvironmentalState
    requires_more_info: bool
    missing_fields: List[str]
    is_final_recommendation: bool
    recommendation_data: Optional[RecommendationResponse] = None
