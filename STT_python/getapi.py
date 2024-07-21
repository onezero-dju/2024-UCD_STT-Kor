import requests

from dotenv import load_dotenv
from os import environ

load_dotenv()

resp = requests.get(
    'https://openapi.vito.ai/v1/transcribe/'+'dCETmlfzS_SuJHWURr6d8Q',
    headers={'Authorization': 'bearer '+ f"{environ.get('TOKEN')}"},
)
resp.raise_for_status()
print(resp.json())
