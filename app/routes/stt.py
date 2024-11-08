from pathlib import Path
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.ml_models.model_handler import ModelHandler
from app.io_data.import_data import read_wav_file
from app.utils.logging import setup_logging
from app.utils.config import load_whisper_params
from dotenv import load_dotenv
import os
import shutil

# 로그 설정 및 환경 변수 로드
setup_logging()
load_dotenv()

router = APIRouter()

# 로그 설정
setup_logging()

# whisper_params.yaml 파일의 경로를 지정
current_dir = Path(__file__).resolve().parent
whisper_params_path = current_dir.parent / 'ml_models' / 'whisper_params.yaml'
whisper_params = load_whisper_params(whisper_params_path)

# ModelHandler 인스턴스 생성
model_handler = ModelHandler(whisper_params)

# 환경 변수에서 AUDIO_FILE_PATH 가져오기
AUDIO_FILE_PATH = os.getenv('AUDIO_FILE_PATH', './audio_files')
audio_file_directory = Path(AUDIO_FILE_PATH)

@router.post("/api/process_audio/")
async def stt(file: UploadFile = File(...)):
    try:
        # 파일명에서 경로 제거
        filename = Path(file.filename).name
        file_path = audio_file_directory / filename
        with file_path.open('wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Uploaded file saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="파일 저장에 실패했습니다.")

    # 파일 검증
    try:
        audio_file_path = read_wav_file(file_path)
    except Exception as e:
        logging.error(f"Invalid audio file: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # 오디오 처리
    try:
        result = model_handler.process_audio(audio_file_path)
    except Exception as e:
        logging.error(f"Failed to process audio: {e}")
        raise HTTPException(status_code=500, detail="오디오 처리에 실패했습니다.")

    # 결과 반환
    return result