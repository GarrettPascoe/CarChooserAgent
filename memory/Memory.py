from typing import Dict
from uuid import uuid4
from schemas.schemas import VehiclePreferences  # adjust import

class PreferenceMemory:
    def __init__(self):
        self._prefs = VehiclePreferences()

    def update(self, new_prefs: VehiclePreferences):
        updated = self._prefs.model_dump()
        for key, value in new_prefs.model_dump().items():
            if value is not None:
                updated[key] = value
        self._prefs = VehiclePreferences(**updated)

    def get(self) -> VehiclePreferences:
        return self._prefs

    def reset(self):
        self._prefs = VehiclePreferences()


class SessionMemoryStore:
    def __init__(self):
        self._sessions: Dict[str, PreferenceMemory] = {}

    def get_or_create(self, session_id: str) -> PreferenceMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = PreferenceMemory()
        return self._sessions[session_id]

    def create_session(self) -> str:
        session_id = str(uuid4())
        self._sessions[session_id] = PreferenceMemory()
        return session_id

    def delete_session(self, session_id: str):
        self._sessions.pop(session_id, None)


memory_store = SessionMemoryStore()