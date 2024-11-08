import json
import logging
from pathlib import Path

def write_json_file(output_file_path, data):
    output_path = Path(output_file_path)
    try:
        # 출력 디렉터리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 결과를 JSON 파일로 저장
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved results to {output_path}.")
    except Exception as e:
        logging.error(f"Failed to save a JSON file: {e}")
        raise e