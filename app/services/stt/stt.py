import whisper
import logging

class TranscriptionService:
    def __init__(self, whisper_params, device='cpu'):
        """
        음성 전사 서비스를 초기화하는 클래스.

        Args:
            whisper_params (dict): Whisper 전사 파라미터.
            device (str): 사용할 디바이스 ('cuda' 또는 'cpu').
        """
        self.whisper_params = whisper_params
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        """
        Whisper 모델을 로드하는 메서드.

        Returns:
            whisper.Model: 로드된 Whisper 모델.
        """
        try:
            model_name = self.whisper_params.get("model", "tiny")
            download_root = self.whisper_params.get("download_root", "./whisper_models")
            logging.info(f"Loading Whisper model '{model_name}'...")
            model = whisper.load_model(model_name, download_root=download_root)
            model.to(self.device)
            logging.info(f"Whisper model '{model_name}' loaded on {self.device}.")
            return model
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            raise e

    def perform_transcription(self, audio_file_path):
        """
        음성 전사를 수행하는 메서드.

        Args:
            audio_file_path (str): 입력 음성 파일의 경로.

        Returns:
            transcription: 전사 결과.
        """
        try:
            logging.info("Performing transcription...")
            transcription = self.model.transcribe(audio_file_path, **self.whisper_params)
            logging.info("Transcription completed.")
            return transcription
        except Exception as e:
            logging.error(f"Failed to perform transcription: {e}")
            raise e
