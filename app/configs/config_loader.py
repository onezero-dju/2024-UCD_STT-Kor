import os
import yaml
import logging

def load_config(config_path):
    """
    YAML 설정 파일을 로드하는 함수.

    Args:
        config_path (str): 설정 파일의 경로.

    Returns:
        dict: 로드된 설정 딕셔너리.
    """
    if not os.path.isfile(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}.")
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise e

def validate_whisper_params(params, required_keys):
    """
    Whisper 파라미터 딕셔너리에 필수 키들이 존재하는지 검증하는 함수.

    Args:
        params (dict): Whisper 파라미터 딕셔너리.
        required_keys (list): 필수 키들의 리스트.
    """
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        logging.error(f"Missing required Whisper parameters: {', '.join(missing_keys)}")
        raise ValueError(f"Missing required Whisper parameters: {', '.join(missing_keys)}")
    
    # 추가적인 유효성 검증
    if not isinstance(params.get("beam_size"), int) or params["beam_size"] <= 0:
        logging.error("Parameter 'beam_size' must be a positive integer.")
        raise ValueError("Parameter 'beam_size' must be a positive integer.")
    
    if not (0.0 <= params.get("temperature", 0.0) <= 1.0):
        logging.error("Parameter 'temperature' must be between 0.0 and 1.0.")
        raise ValueError("Parameter 'temperature' must be between 0.0 and 1.0.")
    
    logging.info("All required Whisper parameters are present and valid.")
