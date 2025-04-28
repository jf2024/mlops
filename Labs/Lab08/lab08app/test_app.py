import requests

# Define the input payload with the correct fields
data = {
    'Crime_Rate': 0.2,
    'Unemployment_Rate': 0.05,
    'Work_Life_Balance': 0.8,
    'Freedom': 0.7
}

url = 'http://127.0.0.1:8000/predict'

# Make the POST request
response = requests.post(url, json=data)

# Print the response JSON which contains the predicted happiness score
print(response.json())
