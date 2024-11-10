# app/ml_models/model_services.py

from pathlib import Path
import logging
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import torch
import warnings
import os

class DiarizationService:
    def __init__(self, access_token: str, device: str):
        self.access_token = access_token
        self.device = device
        self.pipeline = self.load_pipeline()

    def load_pipeline(self) -> Pipeline:
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

    def perform_diarization(self, audio_file_path: Path):
        try:
            logging.info("Performing speaker diarization.")
            diarization = self.pipeline(str(audio_file_path))
            logging.info("Speaker diarization completed.")
            return diarization
        except Exception as e:
            logging.error(f"Failed to perform speaker diarization: {e}")
            raise e

class TranscriptionService:
    def __init__(self, device: str, model_name_or_path: str):
        self.device = device
        self.model = self.load_model(model_name_or_path)

    def load_model(self, model_name_or_path: str) -> WhisperModel:
        try:
            # 디바이스에 따라 compute_type 설정
            if self.device == "cuda":
                compute_type = os.getenv('FWHISPER_COMPUTE_TYPE', 'float16')  # GPU: float16, int8도 가능
            else:
                compute_type = os.getenv('FWHISPER_COMPUTE_TYPE', 'float32')  # CPU: float32 기본값

            logging.info(f"Loading Faster-Whisper model '{model_name_or_path}' on {self.device} with compute_type='{compute_type}'.")
            model = WhisperModel(model_name_or_path, device=self.device, compute_type=compute_type)
            logging.info(f"Faster-Whisper model '{model_name_or_path}' loaded on {self.device}.")
            return model
        except Exception as e:
            logging.error(f"Failed to load Faster-Whisper model: {e}")
            raise e

    def perform_transcription(self, audio_file_path: Path):
        try:
            logging.info(f"Performing transcription on file: {audio_file_path}")

            # FutureWarning 무시 (임시 방편)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                
                # Faster-Whisper의 transcribe 메서드 사용
                segments, info = self.model.transcribe(str(audio_file_path), beam_size=5)

            logging.info("Transcription completed.")
            transcription_dict = {"segments": list(segments), "info": info}
            logging.debug(f"Transcription dict being returned: {transcription_dict}")
            logging.debug(f"Transcription segments type: {type(list(segments))}, info type: {type(info)}")
            return transcription_dict  # generator를 리스트로 변환
        except Exception as e:
            logging.error(f"Failed to perform transcription on file {audio_file_path}: {e}")
            raise e
