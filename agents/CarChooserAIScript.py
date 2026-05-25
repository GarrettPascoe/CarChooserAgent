# %%
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, AgentMiddleware, SummarizationMiddleware
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import os
import json

from schemas.schemas import *
from memory.Memory import *

load_dotenv()


basic_model = ChatOpenAI(model="gpt-4o-mini")


# ------ Section that handles the user's input ------
    
inputParser = PydanticOutputParser(pydantic_object=VehiclePreferences)

extract_prompt = PromptTemplate(
    template=(
        "Extract all vehicle preferences from the following message.\n"
        "Return valid JSON matching the schema.\n\n"
        "Message: {input}\n\n"
        "{format_instructions}"
    ),
    partial_variables={"format_instructions": inputParser.get_format_instructions()},
    input_variables=["input"]
)

extract_chain = extract_prompt | basic_model | inputParser


@tool("extract_preferences")
def extract_prefs_tool(user_input: str):
    """Extract vehicle preferences from user prompt"""
    return extract_chain.invoke({"input": user_input})
    
    
# ------ End of section that handles the user's input ------
    
    
    
# ------ Section that handles the model's stored data ------

class PreferenceMemory:
    def __init__(self):
        self._prefs = VehiclePreferences()

    def update(self, new_prefs: VehiclePreferences):
        """Merge new preferences into memory."""
        updated = self._prefs.model_dump()
        for key, value in new_prefs.model_dump().items():
            if value is not None:
                updated[key] = value
        self._prefs = VehiclePreferences(**updated)

    def get(self) -> VehiclePreferences:
        """Return the full merged preference model."""
        return self._prefs

    def reset(self):
        """Optional: clear memory."""
        self._prefs = VehiclePreferences()
        
        

memory = PreferenceMemory()

@tool("update_preferences")
def update_preferences_tool(session_id: str, **kwargs):
    """Update preferences."""
    memory = memory_store.get_or_create(session_id)
    new_prefs = VehiclePreferences(**kwargs)
    memory.update(new_prefs)
    return "Preferences updated."


@tool("get_preferences")
def get_preferences_tool(session_id: str):
    """Get user preferences."""
    memory = memory_store.get_or_create(session_id)
    return memory.get().model_dump()


# ------ End of section that handles the model's stored data ------


    
# ------ Section that formats the model's response ------
    
outputParser = PydanticOutputParser(pydantic_object=BotResponse)

outputPrompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You must format your final answer as instructed below.
            The answer must be in a JSON format.

            {format_instructions}
            """
        ),
        ("human", "{final_answer}")
    ]
).partial(format_instructions=outputParser.get_format_instructions())


formatter_chain = outputPrompt | basic_model | outputParser


# ------ End of section that formats the model's response ------


SYSTEM_PROMPT = """
You are a helpful assistant that recommends Ford vehicles based on a user's
stated preferences and constraints. Try to recommend the best fit even if
the user's preferences are vague.

Your goals:
- Recommend the most suitable Ford vehicle
- Be concise, accurate, and user-friendly
"""


# Tool to send a query to ChatGPT and handle the response
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"



# Agent constructor
agent = create_agent(
    model=basic_model,  # Default model
    tools=[search, extract_prefs_tool, update_preferences_tool, get_preferences_tool],
    middleware=[],
    system_prompt=SYSTEM_PROMPT
)

def run_agent(messages):
    """
    messages: list[HumanMessage | AIMessage]
    """
    result = agent.invoke({"messages": messages})

    ai_messages = [
        m for m in result["messages"]
        if isinstance(m, AIMessage)
    ]

    final_reply = ai_messages[-1]

    structured: BotResponse = formatter_chain.invoke({
        "final_answer": final_reply.content
    })

    return structured
