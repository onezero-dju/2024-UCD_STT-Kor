import logging

def setup_logging():
    """
    로깅 설정을 초기화하는 함수.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def format_elapsed_time(seconds):
    """
    초 단위의 시간을 'Xm Ys' 형식으로 변환하는 함수.

    Args:
        seconds (float): 소요 시간 (초).

    Returns:
        str: 변환된 시간 문자열 (예: '1m 30s').
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.2f}s"

def format_speaker_label(speaker):
    """
    화자 라벨을 'SPEAKER 01' 형식으로 변환하는 함수.
    예: 'SPEAKER_00' -> 'SPEAKER 00'.

    Args:
        speaker (str): 화자 라벨.

    Returns:
        str: 포맷팅된 화자 라벨.
    """
    if speaker.lower().startswith("speaker_"):
        speaker_num = speaker.split("_")[-1].zfill(2)   # 두 자리 숫자로 포맷 (ex. 1 -> 01)
        return f"SPEAKER {speaker_num}"
    else:
        return f"SPEAKER {speaker}"
