import json
import redis
import logging
from typing import List, Dict, Any

logger = logging.getLogger("nexusai.history")

class HistoryService:
    def __init__(self, host="localhost", port=6379, db=0):
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            # Test connection
            self.client.ping()
            logger.info("✅ Connected to Redis successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.client = None

    def get_history(self, session_id: str = "default") -> List[Dict[str, Any]]:
        if not self.client:
            return []
        try:
            data = self.client.get(f"history:{session_id}")
            if data:
                return json.loads(data)
            return []
        except Exception as e:
            logger.error(f"❌ Failed to get history for {session_id}: {e}")
            return []

    def save_history(self, session_id: str, messages: List[Dict[str, Any]]):
        if not self.client:
            return
        try:
            self.client.set(f"history:{session_id}", json.dumps(messages))
        except Exception as e:
            logger.error(f"❌ Failed to save history for {session_id}: {e}")

    def clear_history(self, session_id: str = "default"):
        if not self.client:
            return
        try:
            self.client.delete(f"history:{session_id}")
        except Exception as e:
            logger.error(f"❌ Failed to clear history for {session_id}: {e}")
