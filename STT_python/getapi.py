# rtzrapi.py 에서 얻은 ID 값을 받아 json 형태로 STT 추출 되는 코드입니다.

import requests

from dotenv import load_dotenv
from os import environ

load_dotenv()

resp = requests.get(
    'https://openapi.vito.ai/v1/transcribe/'+'2pGFIP5TRW-W92R3WU6Ugw',
    headers={'Authorization': 'bearer '+ f"{environ.get('TOKEN')}"},
)
resp.raise_for_status()
print(resp.json())
