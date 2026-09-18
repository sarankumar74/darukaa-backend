import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.main import app
from backend.models.state import EnvironmentalState

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("backend.services.chat.rag_service.retrieve")
@patch("backend.services.chat.llm.invoke")
def test_chat_missing_variables(mock_llm, mock_retrieve):
    # Mocking LLM response for missing variables
    class MockMessage:
        content = '''{
            "response": "Please provide more information.",
            "updated_state": {"soil_organic_carbon": null, "soil_ph": null, "rainfall": null, "land_use": null, "crop_type": null, "region": null},
            "requires_more_info": true,
            "missing_fields": ["soil_organic_carbon", "rainfall", "land_use"],
            "is_final_recommendation": false,
            "recommendation_data": null
        }'''
    mock_llm.return_value = MockMessage()
    mock_retrieve.return_value = []

    req_data = {
        "message": "Biodiversity is declining.",
        "history": [],
        "state": {}
    }
    
    response = client.post("/chat", json=req_data)
    assert response.status_code == 200
    data = response.json()
    assert data["requires_more_info"] == True
    assert "soil_organic_carbon" in data["missing_fields"]
    assert data["is_final_recommendation"] == False

@patch("backend.services.chat.rag_service.retrieve")
@patch("backend.services.chat.llm.invoke")
def test_chat_with_sufficient_variables(mock_llm, mock_retrieve):
    # Mocking LLM response for a full recommendation
    class MockMessage:
        content = '''{
            "response": "Based on your state, here is a recommendation.",
            "updated_state": {"soil_organic_carbon": 0.3, "soil_ph": null, "rainfall": "low", "land_use": "monoculture", "crop_type": "wheat", "region": "semi-arid"},
            "requires_more_info": false,
            "missing_fields": [],
            "is_final_recommendation": true,
            "recommendation_data": {
                "recommendation": "Introduce agroforestry",
                "impacted_metrics": ["soil_organic_carbon", "biodiversity"],
                "scientific_reasoning": "Agroforestry increases SOC by 15-25% in semi-arid regions.",
                "time_horizon": "2-3 years",
                "evidence_source": "FAO Global Soil and Biodiversity Report 2023",
                "confidence_level": "High"
            }
        }'''
    mock_llm.return_value = MockMessage()
    mock_retrieve.return_value = [{"content": "Agroforestry increases SOC...", "source": "FAO Global Soil and Biodiversity Report 2023"}]

    req_data = {
        "message": "What should I do?",
        "history": [{"role": "user", "content": "Biodiversity is declining."}],
        "state": {
            "soil_organic_carbon": 0.3,
            "rainfall": "low",
            "land_use": "monoculture",
            "crop_type": "wheat",
            "region": "semi-arid"
        }
    }
    
    response = client.post("/chat", json=req_data)
    assert response.status_code == 200
    data = response.json()
    
    assert data["requires_more_info"] == False
    assert data["is_final_recommendation"] == True
    assert data["recommendation_data"] is not None
    assert data["recommendation_data"]["recommendation"] == "Introduce agroforestry"
    assert data["recommendation_data"]["confidence_level"] == "High"
