# from pathlib import Path
# import yaml
# import logging

# def load_whisper_params(yaml_file_path):
#     try:
#         yaml_path = Path(yaml_file_path)
#         if not yaml_path.is_file():
#             logging.error(f"Whisper parameters file not found: {yaml_path}")
#             raise FileNotFoundError(f"Whisper parameters file not found: {yaml_path}")
#         with yaml_path.open('r', encoding='utf-8') as f:
#             whisper_params = yaml.safe_load(f)
#         logging.info(f"Whisper parameters loaded from {yaml_path}.")
#         return whisper_params
#     except Exception as e:
#         logging.error(f"Failed to load Whisper parameters: {e}")
#         raise e