# Token을 추출하는 코드입니다.

import requests

resp = requests.post(
    'https://openapi.vito.ai/v1/authenticate',
    data={'client_id': 'tA0L9vCMSPp5OzkmEMGb',
          'client_secret': 'KAMRk6Katwmztw-oF8Yo5qxUQ2OQF6JDQ4hlNutE'}
)
resp.raise_for_status()
print(resp.json())