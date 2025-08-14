import requests
import json
import time

# --- API endpoint ---
URL = "https://rest.synatic.openeasy.io/easyequities/investnow/search"

AUTH_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6Ijc4RDg1MDk0OTZCMkM5OTcyQ0EwQkQ5NDU4NzMxQUJEIiwidHlwIjoiYXQrand0In0.eyJpc3MiOiJodHRwczovL2lkZW50aXR5Lm9wZW5lYXN5LmlvIiwibmJmIjoxNzU1MTg5NTkwLCJpYXQiOjE3NTUxODk1OTAsImV4cCI6MTc1NTE5MTM5MCwiYXVkIjpbImFwaV9nYXRld2F5IiwidXNlcl9wcm9maWxlX2FwaSIsInN0YXRpY19kYXRhX2FwaSIsImludmVzdF9ub3dfYXBpIiwiYXV0b19yZWZpY2FfYXBpIl0sInNjb3BlIjpbIm9wZW5pZCIsInBsYXRmb3JtIiwicHJvZmlsZSIsImFwaV9nYXRld2F5IiwidXNlcl9wcm9maWxlX2FwaSIsInN0YXRpY19kYXRhX2FwaSIsImludmVzdF9ub3dfYXBpIiwiYXV0b19yZWZpY2FfYXBpIl0sImFtciI6WyJwd2QiXSwiY2xpZW50X2lkIjoiNThhZjI1ZDA3YTkzNGM2N2I2NWFhOGMxNTlmMWMxYzIiLCJzdWIiOiJiYmYxYzkwNS1kZjZiLTQxZjktODkwOC04MTMwZWYyMjcxNDUiLCJhdXRoX3RpbWUiOjE3NTUxODkwNzEsImlkcCI6ImxvY2FsIiwic3Vic3lzdGVtaWQiOiIxIiwidGVuYW50aWQiOiIxIiwidXNlcmlkIjoiMjgyNjU2MCIsInNpZCI6IjE5OTY5NjhFQTBFRkI5RjI3NzUxQUVDMDBFMzlDRTk0IiwianRpIjoiREJDOUI2NUNBQkY3M0RCMzg4NzNGNzQzMDI5NzZDMzEifQ.cbEcP975iiW08Ysqy_SOUOvYtanoXqhl65XTx7jdNxddtjkB7HYr6YrT0_4jlTDNeg1JNoYiOxWUSow70LLbKLAK4lnG73oiIQ_JXs77Cq8bSjBXq5xJD0pjzqYCNUiilWHWPhfKUjaqg96O9jCtQ2xQih3avpcM6z0TaYG1eDcZ5t4-EAHxIvHXA2SQP7q9QO3k1k9Y74a0bHRn57IE9hRdPZp634Ri-rfHLByTpWhNT9rybsb3kq0mPijYhm43WfopaQa64hIabYBXO5btFs_ZiEQpaBNoId2OR-tSA5nR4l_b3hSZRjq4FWHoG10GC_8mzjqpUfiy-yLx5GsKMA"

# --- Headers ---
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH_TOKEN}"
}

all_results = []
page = 1

while True:
    payload = {
        "searchValue": "",
        "account_filter": "ALL",
        "category": "equitiesexpanded",  # changed category
        "page": page
    }

    print(f"Fetching page {page}...")
    response = requests.post(URL, json=payload, headers=HEADERS)

    if response.status_code != 200:
        print(f"Request failed: {response.status_code} {response.text}")
        break

    data = response.json()

    # EasyEquities uses "results" or "instruments"
    results = data.get("results") or data.get("instruments")

    if not results:
        print("No more data. Stopping.")
        break

    all_results.extend(results)
    page += 1

    # Avoid hammering the server
    time.sleep(1)

# Save all commodities to JSON
with open("equities.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"Saved {len(all_results)} items to commodities.json")
