import httpx
import json

def test_fetch_mail():
    url = "http://localhost:8000/mail/fetch"
    payload = {
        "email": "test@example.com",
        "imap_server": "imap.example.com",
        "password": "wrong_password"
    }
    try:
        response = httpx.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_fetch_mail()
