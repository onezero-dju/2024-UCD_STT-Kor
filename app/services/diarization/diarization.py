import torch
from pyannote.audio import Pipeline
import logging

class DiarizationService:
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
