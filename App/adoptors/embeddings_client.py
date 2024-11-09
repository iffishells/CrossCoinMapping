from dotenv import dotenv_values
import os
import requests
config = dotenv_values(os.path.join('.env'))
def get_embeddings(texts=None):
        headers = {
            "Content-Type": "application/json",
            "api-key": config['API_KEY']
        }
        data = {
            "input": texts,
            "model": "text-embedding-ada-002"  # Change as needed based on your deployment
        }

        response = requests.post(config['ENDPOINT'], headers=headers, json=data)

        if response.status_code == 200:
            return response.json()['data']
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None