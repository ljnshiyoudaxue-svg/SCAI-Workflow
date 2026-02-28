import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:14b"

class LocalLLMError(Exception):
    pass


def call_local_llm(prompt, temperature=0.2, max_tokens=2048):
    """
    本地 Ollama LLM 调用接口
    可直接替换原有 OpenAI / DeepSeek API
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
    except Exception as e:
        raise LocalLLMError(f"Ollama connection failed: {e}")

    if response.status_code != 200:
        raise LocalLLMError(f"Ollama error: {response.text}")

    result = response.json()

    if "response" not in result:
        raise LocalLLMError("Invalid Ollama response format")

    return result["response"]