from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal
from enum import Enum

class Message(BaseModel):
    type: str   # "human" or "ai"
    content: str

class AgentRequest(BaseModel):
    session_id: str
    messages: List[Message]
    
class MakeEnum(str, Enum):
    ford = "Ford"
    
# Pydantic Model for vehicle attributes
class VehiclePreferences(BaseModel):
    manufacturer: MakeEnum = Field(default=MakeEnum.ford, description="Manufacturer, must be Ford")
    model: Optional[str] = None
    body_style:  Optional[str] = None
    price_range:  Optional[str] = None
    mileage:  Optional[str] = None
    safety_rating:  Optional[str] = None
    color:  Optional[str] = None
    storage_space:  Optional[str] = None
    seat_capacity:  Optional[str] = None
    comfort:  Optional[str] = None
    luxury:  Optional[str] = None
    special_features: Optional[str] = None
    unwanted_features: Optional[str] = None
    
# Pydantic Model to validate user input
# class UserQuery(BaseModel):
#    intent: str = Field(..., description="A description of what the user is looking for.")
#    preferences: list[VehiclePreferences] = Field(
#        default_factory=list,
#        description="List of parsed structured preferences about the Ford vehicle."
#    )
    
    
# Pydantic model to structure model response
class BotResponse(BaseModel):
        message: str = Field(..., description="The chatbot response")

        link: Optional[str] = Field(None, description="URL of a Ford vehicle page. Only present when recommending a vehicle.")

        action_required: Optional[Literal["confirm", "clarify", "provide_info"]] = None

        @model_validator(mode="after")
        def validate_state_logic(self):
        # If confirming a selection -> must include link
            if self.action_required == "confirm" and not self.link:
                raise ValueError("confirm responses must include a vehicle link")

            # If clarifying -> must NOT include link
            if self.action_required == "clarify" and self.link:
                raise ValueError("clarification responses cannot include a link")

            return self