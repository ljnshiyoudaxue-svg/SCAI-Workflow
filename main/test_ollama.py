import requests
import json

def test_ollama():
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "deepseek-r1:14b",
        "prompt": "请用一句话解释什么是裂缝。",
        "stream": False
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✅ 调用成功！\n")
        print(result["response"])
    else:
        print("❌ 调用失败")
        print(response.text)


if __name__ == "__main__":
    test_ollama()