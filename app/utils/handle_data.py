import os
import logging

def read_wav_file(audio_file_path):
    """
    .wav 파일을 읽는 함수.

    Args:
        audio_file_path (str): 입력 음성 파일의 경로.

    Returns:
        str: 읽은 파일의 내용 (필요 시 수정).
    """
    if not os.path.isfile(audio_file_path):
        logging.error(f"Audio file not found: {audio_file_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    logging.info(f"Audio file {audio_file_path} is ready for processing.")
    return audio_file_path

def write_txt_file(output_file_path, lines):
    """
    전사 결과를 텍스트 파일로 저장하는 함수.

    Args:
        output_file_path (str): 출력 파일의 경로.
        lines (list): 저장할 텍스트 라인 리스트.
    """
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        logging.info(f"Saved results to {output_file_path}.")
    except Exception as e:
        logging.error(f"Failed to save a text file: {e}")
        raise e
