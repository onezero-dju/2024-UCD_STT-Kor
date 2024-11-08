from pathlib import Path
import logging
from pyannote.audio import Pipeline
import whisper

class DiarizationService:
    def __init__(self, access_token, device):
        self.access_token = access_token
        self.device = device
        self.pipeline = self.load_pipeline()

    def load_pipeline(self):
        try:
            logging.info("Loading speaker diarization pipeline.")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.access_token
            )
            logging.info("Diarization pipeline loaded.")
            return pipeline
        except Exception as e:
            logging.error(f"Failed to load speaker diarization pipeline: {e}")
            raise e

    def perform_diarization(self, audio_file_path):
        try:
            logging.info("Performing speaker diarization.")
            diarization = self.pipeline(str(audio_file_path))
            logging.info("Speaker diarization completed.")
            return diarization
        except Exception as e:
            logging.error(f"Failed to perform speaker diarization: {e}")
            raise e

class TranscriptionService:
    def __init__(self, whisper_params, device):
        self.whisper_params = whisper_params
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        try:
            model_name_or_path = self.whisper_params.get("model", "tiny")
            model_path = Path(model_name_or_path)
            if model_path.is_file():
                # 커스텀 모델 경로 사용
                logging.info(f"Loading custom Whisper model from '{model_path}'.")
                model = whisper.load_model(str(model_path), device=self.device)
            else:
                # 기본 모델 사용
                download_root = self.whisper_params.get("download_root", "./whisper_models")
                download_root_path = Path(download_root)
                logging.info(f"Loading Whisper model '{model_name_or_path}' from '{download_root_path}'.")
                model = whisper.load_model(model_name_or_path, download_root=str(download_root_path), device=self.device)
            logging.info(f"Whisper model '{model_name_or_path}' loaded on {self.device}.")
            return model
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            raise e

    def perform_transcription(self, audio_file_path):
        try:
            logging.info(f"Performing transcription on file: {audio_file_path}")

            # 모델 관련 파라미터 제거
            whisper_params = self.whisper_params.copy()
            whisper_params.pop("model", None)
            whisper_params.pop("download_root", None)

            # 단어별 타임스탬프 활성화
            whisper_params['word_timestamps'] = True

            transcription = self.model.transcribe(str(audio_file_path), **whisper_params)
            logging.info("Transcription completed.")
            return transcription
        except Exception as e:
            logging.error(f"Failed to perform transcription on file {audio_file_path}: {e}")
            raise e
