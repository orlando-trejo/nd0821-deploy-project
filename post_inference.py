import requests

data = {
    "age": 25,
    "workclass": "Private",
    "fnlgt": 226802,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Never-married",
    "occupation": "Machine-op-inspct",
    "relationship": "Own-child",
    "race": "Black",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

url = "https://nd0821-deploy-project.onrender.com/predict"

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response:", response.json())

