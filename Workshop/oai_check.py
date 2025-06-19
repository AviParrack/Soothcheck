import os
import openai
from openai import OpenAI
import requests


def check_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return False

    print(f"✓ Found API key: {api_key[:8]}...{api_key[-4:]}")

    # First test raw requests
    try:
        print("Testing raw requests...")
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        print(f"Raw request status: {response.status_code}")
    except Exception as e:
        print(f"Raw requests failed: {e}")
        return False

    # Then test OpenAI client
    try:
        print("Testing OpenAI client...")
        client = OpenAI(api_key=api_key, timeout=10.0)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        print("✅ API key is valid!")
        print(f"Test response: {response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"OpenAI client error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    check_openai_api_key()
