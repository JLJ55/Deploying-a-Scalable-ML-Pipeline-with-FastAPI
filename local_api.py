import json
import requests

# Send a GET request to the root URL
url = "http://127.0.0.1:8000"
r = requests.get(url)

# Print the status code
print("GET Status Code:", r.status_code)

# Print the welcome message
print("GET Response Message:", r.json())

# Data for the POST request
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

r = requests.post(f"{url}/data/", json=data)

# Print the status code
print("POST Status Code:", r.status_code)

# Print the result
print("POST Response:", r.json())
