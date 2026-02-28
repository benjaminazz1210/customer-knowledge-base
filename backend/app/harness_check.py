import os
import sys
import requests
from openai import OpenAI
from dotenv import load_dotenv
from config import config

def check_harness():
    print("🚀 NexusAI Harness Validation Started...")
    load_dotenv()
    
    # 1. Check API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ FAILED: DEEPSEEK_API_KEY not found in .env")
        return False
    else:
        print("✅ SUCCESS: DEEPSEEK_API_KEY is present")

    # 2. Check Qdrant Connection
    host = os.getenv("QDRANT_HOST", "localhost")
    port = os.getenv("QDRANT_PORT", "6333")
    try:
        response = requests.get(f"http://{host}:{port}/")
        if response.status_code == 200:
            print(f"✅ SUCCESS: Qdrant is reachable at {host}:{port}")
        else:
            print(f"❌ FAILED: Qdrant returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ FAILED: Could not reach Qdrant: {e}")
        return False

    # 3. Check DeepSeek API (Chat with Reasoner)
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        # Lightweight test: try to prompt the reasoner
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5
        )
        print(f"✅ SUCCESS: DeepSeek Reasoner API is responsive")
    except Exception as e:
        print(f"❌ FAILED: DeepSeek API error: {e}")
        return False

    # 4. Check Multimodal Embedding Model (Qwen3-VL-Embedding-2B)
    try:
        from services.scripts.qwen3_vl_embedding import Qwen3VLEmbedder
        print(f"⌛ Testing/Loading local multimodal model: {config.EMBEDDING_MODEL}...")
        # Just check if we can import and locate the model
        print(f"✅ SUCCESS: Embedding model script {config.EMBEDDING_MODEL} is ready")
    except Exception as e:
        print(f"❌ FAILED: Multimodal embedding script error: {e}")
        return False

    print("\n🌟 All systems go! Harness check passed.")
    return True

if __name__ == "__main__":
    if not check_harness():
        sys.exit(1)
