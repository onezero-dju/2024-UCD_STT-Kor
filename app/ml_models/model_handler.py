# app/ml_models/model_handler.py

import logging
from pathlib import Path
from datetime import datetime
from app.ml_models.model_services import DiarizationService, TranscriptionService
from dotenv import load_dotenv
import os
import json
import torch

class ModelHandler:
    def __init__(self):
        # 환경 변수 로드
        load_dotenv()
        # 환경 변수 가져오기
        AUDIO_FILE_PATH = os.getenv('AUDIO_FILE_PATH', './audio_files')
        OUTPUT_DIRECTORY = os.getenv('OUTPUT_DIRECTORY', './outputs')
        HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
        FWHISPER_MODEL = os.getenv('FWHISPER_MODEL', 'large-v2')
        # FWHISPER_COMPUTE_TYPE은 TranscriptionService에서 자동 설정
        
        # 경로를 Path 객체로 변환
        self.audio_file_path = Path(AUDIO_FILE_PATH)
        self.output_directory = Path(OUTPUT_DIRECTORY)

        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diarization_service = DiarizationService(HUGGINGFACE_TOKEN, self.device)
        self.transcription_service = TranscriptionService(self.device, model_name_or_path=FWHISPER_MODEL)  # FWHISPER_MODEL 전달

        # 출력 디렉터리 생성
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def process_audio(self, audio_file_path: Path):
        audio_path = Path(audio_file_path)

        # 화자 분리 수행
        diarization = self.diarization_service.perform_diarization(audio_path)

        # 전체 오디오 파일 전사 수행 (타임스탬프 포함)
        transcription = self.transcription_service.perform_transcription(audio_path)

        # transcription 객체의 타입과 내용 로깅
        logging.debug(f"Transcription type: {type(transcription)}")
        logging.debug(f"Transcription contents: {transcription}")

        # 결과 통합 및 JSON 형식으로 변환
        result = self.integrate_results(diarization, transcription, audio_path)

        # 결과를 OUTPUT_DIRECTORY에 저장
        output_file_name = f"{audio_path.stem}.json"
        output_file_path = self.output_directory / output_file_name
        self.save_result(output_file_path, result)

        return result

    def integrate_results(self, diarization, transcription, audio_path: Path):
        if not isinstance(transcription, dict):
            logging.error(f"Expected transcription to be a dict, got {type(transcription)}")
            raise ValueError(f"Expected transcription to be a dict, got {type(transcription)}")

        segments = []
        segment_count = 0

        # Faster-Whisper 전사 결과에서 세그먼트 추출
        segments_transcription = transcription.get('segments', [])

        logging.info(f"Total transcription segments: {len(segments_transcription)}")

        # 세그먼트별로 매핑
        for segment in diarization.itertracks(yield_label=True):
            segment_count += 1
            start_time = segment[0].start
            end_time = segment[0].end
            speaker = segment[2]

            logging.info(f"Processing diarization segment {segment_count}: Start {start_time}, End {end_time}, Speaker {speaker}")

            # 해당 세그먼트에 속하는 전사 세그먼트 추출
            mapped_segments = [
                s for s in segments_transcription
                if s.start >= start_time and s.end <= end_time
            ]

            # 전사 텍스트 결합
            text = ' '.join([s.text for s in mapped_segments])

            segments.append({
                "start": start_time,
                "end": end_time,
                "text": text,
                "speaker": speaker,
                "confidence": None  # Faster-Whisper는 confidence 점수를 기본적으로 제공하지 않음
            })

            logging.debug(f"Segment {segment_count} mapped text: {text}")

        # 전체 전사 텍스트 (Faster-Whisper는 전체 텍스트를 별도로 제공하지 않음)
        transcription_text = ' '.join([s.text for s in segments_transcription])

        # 메타데이터 생성
        info = transcription.get('info')
        language = info.language if info else 'unknown'

        metadata = {
            "language": language,
            "audio_file": audio_path.name,
            "date": datetime.now().strftime("%Y-%m-%d")
        }

        result = {
            "transcription": transcription_text,
            "segments": segments,
            "metadata": metadata
        }
        return result

    def save_result(self, output_file_path: Path, result: dict):
        try:
            # 출력 디렉터리 생성
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            # 결과를 JSON 파일로 저장
            with output_file_path.open('w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            logging.info(f"Saved result to {output_file_path}.")
        except Exception as e:
            logging.error(f"Failed to save result to {output_file_path}: {e}")
            raise e
