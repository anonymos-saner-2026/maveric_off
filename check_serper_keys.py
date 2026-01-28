
import requests
import json
import time

KEYS = [
    "b2548827d405252eaafecf6096553e3a961df350",
    "e6ff359453fd0924d7baae4ed0ae77167b6d91da",
    "be246072ad5bd57534d6997419b6360e7789b0da",
    "c3314ee6d8971521f3966c265935973fd116797b",
    "a0c4f224c72807001f5fa95a62c648eac4cc14ad",
    "92635770b58623791fd14b56600ce0c89c0632a2",
    "86752f76b665da94fa68efe670d6f5c07bdc6434",
    "3cefc8851d21371942d5cfb80f23e9648a667de4"  # Default in config
]

def check_key(api_key):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": "test query", "num": 1})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        status = response.status_code
        try:
            data = response.json()
        except:
            data = {}
            
        return status, data
    except Exception as e:
        return -1, str(e)

print(f"Checking {len(KEYS)} Serper keys...\n")

valid_count = 0
for i, key in enumerate(KEYS):
    masked_key = f"{key[:6]}...{key[-4:]}"
    print(f"[{i+1}/{len(KEYS)}] Checking {masked_key} ... ", end="", flush=True)
    
    status, data = check_key(key)
    
    if status == 200:
        print("✅ OK")
        valid_count += 1
    elif status == 403:
        print("❌ 403 Forbidden (Invalid or Quota Exceeded)")
    elif status == 429:
        print("❌ 429 Too Many Requests")
    else:
        print(f"⚠️ Status {status}: {data}")
    
    # Small delay to avoid rate limits if any
    time.sleep(0.5)

print(f"\nSummary: {valid_count}/{len(KEYS)} keys are working.")
