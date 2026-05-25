from fastapi import FastAPI, UploadFile, File
from agents.CarChooserAIScript import *
from schemas.schemas import *
from memory.Memory import *
import shutil
from ml.ImageClassifierModel import predict_image
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# setting up CORS for image submissions from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.garrettpascoe.com/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    type: str
    content: str

class AgentRequest(BaseModel):
    messages: list[Message]
    session_id: str
    
# CarChooserAgent Section

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

# Image Classifier Section

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predict_image(temp_path)
    
    os.remove(temp_path)

    return {"prediction": prediction}