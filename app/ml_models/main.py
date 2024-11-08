import sys
from pathlib import Path
import logging
import os
from dotenv import load_dotenv

# 필요한 모듈 임포트
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from ml_models.model_handler import ModelHandler
from utils.config import load_whisper_params
from io_data.import_data import read_wav_file

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s')

def main():
    # 로그 설정
    setup_logging()
    
    # 환경 변수 로드
    load_dotenv(dotenv_path=project_root / '.env')

    # 환경 변수 가져오기
    AUDIO_FILE_PATH = os.getenv('AUDIO_FILE_PATH', './audio_files')
    OUTPUT_DIRECTORY = os.getenv('OUTPUT_DIRECTORY', './outputs')

    # 테스트할 오디오 파일의 이름 설정
    audio_file_name = 'UCD_TEST.wav'  # 실제 테스트할 파일명으로 변경

    # 오디오 파일 경로 생성
    audio_file_path = Path(AUDIO_FILE_PATH) / audio_file_name

    # 오디오 파일 존재 여부 확인
    if not audio_file_path.is_file():
        logging.error(f"Audio file not found: {audio_file_path}")
        return

    # Whisper 파라미터 로드
    whisper_params_path = current_dir / 'whisper_params.yaml'
    whisper_params = load_whisper_params(whisper_params_path)

    # ModelHandler 초기화
    model_handler = ModelHandler(whisper_params)

    # 오디오 파일 검증 및 처리
    try:
        validated_audio_file_path = read_wav_file(audio_file_path)
        result = model_handler.process_audio(validated_audio_file_path)
        # 결과 출력
        # print(result)
    except Exception as e:
        logging.error(f"Failed to process audio: {e}")

if __name__ == '__main__':
    main()