import torch
import logging
from pyannote.audio import Pipeline
import whisper

class Diarization:
    def __init__(self, access_token, device='cpu'):
        """
        화자 다이어리제이션 서비스를 초기화하는 클래스.

        Args:
            access_token (str): Hugging Face 액세스 토큰.
            device (str): 사용할 디바이스 ('cuda' 또는 'cpu').
        """
        self.access_token = access_token
        self.device = device
        self.pipeline = self.load_pipeline()

    def load_pipeline(self):
        """
        화자 다이어리제이션 파이프라인을 로드하는 메서드.

        Returns:
            Pipeline: 로드된 다이어리제이션 파이프라인.
        """
        try:
            logging.info("Loading speaker diarization pipeline...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.access_token
            )
            pipeline.to(self.device)
            logging.info(f"Diarization pipeline loaded on {self.device}.")
            return pipeline
        except Exception as e:
            logging.error(f"Failed to load speaker diarization pipeline: {e}")
            raise e

    def perform_diarization(self, audio_file_path):
        """
        화자 다이어리제이션을 수행하는 메서드.

        Args:
            audio_file_path (str): 입력 음성 파일의 경로.

        Returns:
            diarization: 화자 다이어리제이션 결과.
        """
        try:
            logging.info("Performing speaker diarization...")
            diarization = self.pipeline(audio_file_path)
            logging.info("Speaker diarization completed.")
            return diarization
        except Exception as e:
            logging.error(f"Failed to perform speaker diarization: {e}")
            raise e

class SpeechToText:
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
