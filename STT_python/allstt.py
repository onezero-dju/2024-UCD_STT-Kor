# 한번에 TOKEN, ID 값을 받아 STT가 구현되는 코드입니다.

import json
import requests
from dotenv import load_dotenv, set_key
from os import environ
import time

# 함수 정의

def update_token(client_id, client_secret):
    resp = requests.post(
        'https://openapi.vito.ai/v1/authenticate',
        data={'client_id': client_id, 'client_secret': client_secret}
    )
    resp.raise_for_status()
    token = resp.json().get('access_token')
    
    # .env 파일 업데이트
    dotenv_path = '.env'
    set_key(dotenv_path, 'TOKEN', token)

    return token

def get_transcription_id(token, audio_file, config):
    resp = requests.post(
        'https://openapi.vito.ai/v1/transcribe',
        headers={'Authorization': 'bearer ' + token},
        data={'config': json.dumps(config)},
        files={'file': open(audio_file, 'rb')}
    )
    resp.raise_for_status()
    return resp.json().get('id')

def get_transcription_result(token, transcription_id):
    resp = requests.get(
        f'https://openapi.vito.ai/v1/transcribe/{transcription_id}',
        headers={'Authorization': 'bearer ' + token},
    )
    resp.raise_for_status()
    return resp.json()

def wait_for_transcription_result(token, transcription_id, wait_time=5, max_retries=20):
    retries = 0
    while retries < max_retries:
        result = get_transcription_result(token, transcription_id)
        if result.get('status') == 'completed':
            return result
        time.sleep(wait_time)
        retries += 1
    raise TimeoutError("STT 결과를 가져오는 데 실패했습니다.")

# 메인 실행 코드

def main():
    load_dotenv()
    
    # .env 파일에서 API 클라이언트 ID와 PW 읽기
    client_id = environ.get('CLIENT_ID')
    client_secret = environ.get('CLIENT_SECRET')
    
    # 새로운 토큰 받아서 저장
    token = update_token(client_id, client_secret)
    
    # 오디오 파일 경로
    audio_file = '/Users/minhyeok/Desktop/PROJECT/UCD_PROJECT/2024-UCD_STT-Kor/STT_python/UCD_TEST2.m4a'
    
    # 구성 설정
    config = {
        "use_diarization": True,
        "diarization": {"spk_count": 5},
        "use_itn": True,
        "use_disfluency_filter": True,
    }
    
    # 변동되는 ID 추출
    transcription_id = get_transcription_id(token, audio_file, config)
    print(f'Transcription ID: {transcription_id}')
    
    # STT 결과를 기다리고 가져오기
    result = wait_for_transcription_result(token, transcription_id)
    print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()