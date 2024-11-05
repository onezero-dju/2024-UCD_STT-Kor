from pathlib import Path
import tempfile
import torch
from datetime import datetime
from app.ml_models.model_services import DiarizationService, TranscriptionService
from app.io_data.export_data import write_json_file
from dotenv import load_dotenv
import os

class ModelHandler:
    def __init__(self, whisper_params):
        # 프로젝트 루트 디렉터리 경로
        # project_root = Path(__file__).resolve().parent.parent.parent

        # .env 파일 로드
        load_dotenv()
        # load_dotenv(dotenv_path=project_root / '.env')
        
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

        # 전사 수행
        transcription = self.transcription_service.perform_transcription(audio_path)

        # 결과 통합 및 JSON 형식으로 변환
        result = self.integrate_results(diarization, transcription, audio_path)

        # 결과를 OUTPUT_DIRECTORY에 저장
        output_file_name = f"{audio_path.stem}.json"
        output_file_path = self.output_directory / output_file_name
        write_json_file(output_file_path, result)

        return result

    def integrate_results(self, diarization, transcription, audio_path):
        # 세그먼트별로 오디오를 추출하여 전사 결과 매핑
        segments = []
        for segment in diarization.itertracks(yield_label=True):
            start_time = segment[0].start
            end_time = segment[0].end
            speaker = segment[2]

            # 세그먼트별 오디오 추출
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                tmp_audio_path = Path(tmp_audio_file.name)
                command = f'ffmpeg -i "{audio_path}" -ss {start_time} -to {end_time} -c copy "{tmp_audio_path}" -loglevel error'
                os.system(command)

                # 세그먼트 오디오 전사
                segment_transcription = self.transcription_service.perform_transcription(tmp_audio_path)
                text = segment_transcription.get('text', '')
                confidence = segment_transcription.get('confidence', 0.0)

            # 임시 파일 삭제
            tmp_audio_path.unlink()

            segments.append({
                "start": start_time,
                "end": end_time,
                "text": text,
                "speaker": speaker,
                "confidence": confidence
            })

        # 전체 전사 텍스트
        transcription_text = transcription.get('text', '')

        # 메타데이터 생성
        metadata = {
            "language": transcription.get('language', 'ko'),
            "audio_file": audio_path.name,
            "date": datetime.now().strftime("%Y-%m-%d")
        }

        result = {
            "transcription": transcription_text,
            "segments": segments,
            "metadata": metadata
        }
        return result