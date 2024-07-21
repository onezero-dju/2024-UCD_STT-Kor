import json
import requests

from dotenv import load_dotenv
from os import environ

load_dotenv()

config = {
"use_diarization": True,
  "diarization": {
    "spk_count": 5
  },
    "use_itn": True,
    "use_disfluency_filter": True,

}
resp = requests.post(
    'https://openapi.vito.ai/v1/transcribe',
    headers={'Authorization': 'bearer '+ f"{environ.get('TOKEN')}"},
    data={'config': json.dumps(config)},
    files={'file': open('UCD_TEST.m4a', 'rb')}
)
resp.raise_for_status()
print(resp.json())