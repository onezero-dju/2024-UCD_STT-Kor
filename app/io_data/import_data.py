from pathlib import Path
import logging

def read_wav_file(audio_file_path):
    audio_path = Path(audio_file_path)
    # 입력 데이터 디렉터리 탐색
    if not audio_path.is_file():
        logging.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    # .wav 형식자 검토
    if audio_path.suffix.lower() != '.wav':
        logging.error(f"Unsupported file type: {audio_path}. Only .wav files are supported.")
        raise ValueError(f"Unsupported file type: {audio_path}. Only .wav files are supported.")

    logging.info(f"Audio file {audio_path} is ready for processing.")
    return audio_path