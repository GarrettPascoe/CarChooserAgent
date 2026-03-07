from fastapi import FastAPI
from CarChooserAIScript import *
from schemas import *
from Memory import *

app = FastAPI()

class Message(BaseModel):
    type: str
    content: str

class AgentRequest(BaseModel):
    messages: list[Message]
    session_id: str

@app.post("/run-agent", response_model=BotResponse)
async def run_agent_endpoint(payload: AgentRequest):

    lc_messages = []
    for m in payload.messages:
        if m.type == "human":
            lc_messages.append(HumanMessage(content=m.content))
        else:
            lc_messages.append(AIMessage(content=m.content))

    result = agent.invoke({"messages": lc_messages,
                           "session_id": payload.session_id
                           })

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    final_reply = ai_messages[-1]

    return {"message": final_reply.content}

@app.post("/create-session")
async def create_session():
    session_id = memory_store.create_session()
    return {"session_id": session_id}