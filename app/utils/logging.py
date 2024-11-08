import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def format_elapsed_time(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.2f}s"

def format_speaker_label(speaker):
    if speaker.lower().startswith("speaker_"):
        speaker_num = speaker.split("_")[-1].zfill(2)   # 두 자리 숫자로 포맷 (ex. 1 -> 01)
        return f"SPEAKER {speaker_num}"
    else:
        return f"SPEAKER {speaker}"
