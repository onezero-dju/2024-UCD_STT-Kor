import logging
import torch
from pathlib import Path
from datetime import datetime
from app.ml_models.model_services import DiarizationService, TranscriptionService
from dotenv import load_dotenv
import os
import json

class ModelHandler:
    def __init__(self, whisper_params):
        # 환경 변수 로드
        load_dotenv()
        # 환경 변수 가져오기
        AUDIO_FILE_PATH = os.getenv('AUDIO_FILE_PATH', './audio_files')
        OUTPUT_DIRECTORY = os.getenv('OUTPUT_DIRECTORY', './outputs')
        HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

        # 경로를 Path 객체로 변환
        self.audio_file_path = Path(AUDIO_FILE_PATH)
        self.output_directory = Path(OUTPUT_DIRECTORY)

        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diarization_service = DiarizationService(HUGGINGFACE_TOKEN, self.device)
        self.transcription_service = TranscriptionService(whisper_params, self.device)

        # 출력 디렉터리 생성
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def process_audio(self, audio_file_path):
        audio_path = Path(audio_file_path)

        # 화자 분리 수행
        diarization = self.diarization_service.perform_diarization(audio_path)

        # 전체 오디오 파일 전사 수행 (타임스탬프 포함)
        transcription = self.transcription_service.perform_transcription(audio_path)

        # 결과 통합 및 JSON 형식으로 변환
        result = self.integrate_results(diarization, transcription, audio_path)

        # 결과를 OUTPUT_DIRECTORY에 저장
        output_file_name = f"{audio_path.stem}.json"
        output_file_path = self.output_directory / output_file_name
        self.save_result(output_file_path, result)

        return result

    def integrate_results(self, diarization, transcription, audio_path):
        segments = []
        segment_count = 0

        # Whisper 전사 결과에서 단어별 타임스탬프 추출
        words = []
        for segment in transcription['segments']:
            for word_info in segment['words']:
                words.append({
                    'start': word_info['start'],
                    'end': word_info['end'],
                    'text': word_info['word']
                })

        # 세그먼트별로 전사 결과 매핑
        for segment in diarization.itertracks(yield_label=True):
            segment_count += 1
            start_time = segment[0].start
            end_time = segment[0].end
            speaker = segment[2]

            logging.info(f"Processing segment {segment_count}: Start {start_time}, End {end_time}, Speaker {speaker}")

            # 해당 세그먼트에 속하는 단어들 추출
            segment_words = [w['text'] for w in words if w['start'] >= start_time and w['end'] <= end_time]
            text = ' '.join(segment_words)

            segments.append({
                "start": start_time,
                "end": end_time,
                "text": text,
                "speaker": speaker,
                "confidence": None  # 필요 시 계산
            })

        # 전체 전사 텍스트
        transcription_text = transcription.get('text', '')

        # 메타데이터 생성
        metadata = {
            "language": transcription.get('language', 'unknown'),
            "audio_file": audio_path.name,
            "date": datetime.now().strftime("%Y-%m-%d")
        }

        result = {
            "transcription": transcription_text,
            "segments": segments,
            "metadata": metadata
        }
        return result

    def save_result(self, output_file_path, result):
        # 출력 디렉터리 생성
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        # 결과를 JSON 파일로 저장
        with output_file_path.open('w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved result to {output_file_path}.")
