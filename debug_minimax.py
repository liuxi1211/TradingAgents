"""Quick MiniMax API diagnostic — run with: python debug_minimax.py"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def check_env():
    """Check which MiniMax env vars are set."""
    keys = {
        "MINIMAX_API_KEY": "Global (api.minimax.io)",
        "MINIMAX_CN_API_KEY": "China (api.minimaxi.com)",
    }
    print("=== Environment Variables ===")
    for var, label in keys.items():
        val = os.environ.get(var, "")
        if val:
            print(f"  {var} ({label}): {val[:8]}...{val[-4:]}  (len={len(val)})")
        else:
            print(f"  {var} ({label}): NOT SET")
    print()


def test_api(base_url, api_key, model, label):
    """Send a minimal request and print the raw response."""
    import httpx
    import json

    print(f"=== Testing {label} ===")
    print(f"  URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  Key: {api_key[:8]}...{api_key[-4:]}")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in 5 words."}],
        "max_tokens": 50,
    }

    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"  Response: {content[:100]}")
        else:
            print(f"  Error: {resp.text[:500]}")
    except Exception as e:
        print(f"  Exception: {e}")
    print()


def main():
    check_env()

    cn_key = os.environ.get("MINIMAX_CN_API_KEY", "")
    global_key = os.environ.get("MINIMAX_API_KEY", "")

    models_to_test = [
        "MiniMax-M2.7-highspeed",
        "MiniMax-M2.7",
        "MiniMax-M2.5",
    ]

    # Test China endpoint if key is available
    if cn_key:
        for model in models_to_test:
            test_api(
                "https://api.minimaxi.com/v1",
                cn_key,
                model,
                f"China / {model}",
            )
    else:
        print("Skipping China endpoint — MINIMAX_CN_API_KEY not set.\n")

    # Test Global endpoint if key is available
    if global_key:
        for model in models_to_test:
            test_api(
                "https://api.minimax.io/v1",
                global_key,
                model,
                f"Global / {model}",
            )
    else:
        print("Skipping Global endpoint — MINIMAX_API_KEY not set.\n")


if __name__ == "__main__":
    main()
