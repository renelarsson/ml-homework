# test the FastAPI endpoint:
# 1. Sends a POST request to the /predict endpoint with the given customer data.
# 2. Prints whether the lead is likely to convert based on the response.

import requests

url = 'http://localhost:9696/predict'

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client)

if response.status_code == 200:
    predictions = response.json()
    prob = predictions['conversion_probability']
    print(f"Conversion probability: {prob:.3f}")
    if predictions['convert']:
        print('Lead is likely to convert, follow up')
    else:
        print('Lead is not likely to convert')
else:
    print(f"Failed to get a response: {response.status_code}")