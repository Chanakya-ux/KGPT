import requests
import time

# Replace with your actual deployed API URL
API_URL = "https://kgpt-1.onrender.com/query"

def ask_question(question: str):
    payload = {"question": question}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        answer = response.json().get("answer", "No answer returned.")
        print("\n=== Answer ===\n")
        print(answer)
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

if __name__ == "__main__":
    user_question = input("Ask something: ")
    xy = time.time()
    ask_question(user_question)
    print(f"\nRequest completed in {time.time() - xy:.2f} seconds")
