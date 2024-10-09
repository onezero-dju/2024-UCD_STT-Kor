import whisper
from pyannote.audio import Pipeline
from pymongo import MongoClient
import numpy as np
import tempfile
import wave
import torch

# MongoDB 설정
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database']
collection = db['your_collection']

# Whisper 모델과 pyannote.audio 파이프라인 불러오기
whisper_model = whisper.load_model("large-v3")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")

# 음성 데이터 가져오기
def get_audio_data(document_id):
    document = collection.find_one({'_id': document_id})
    return document['audio_data']  # audio_data 필드에 음성 데이터가 있다고 가정

# 화자 분리 및 STT 수행
def process_audio(document_id):
    audio_data = get_audio_data(document_id)
    
    # 임시 파일에 음성 데이터 저장
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        with wave.open(tmp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)
        tmp_file.flush()

        # 화자 다이어리제이션 수행
        diarization = diarization_pipeline(tmp_file.name)
        
        # Whisper 모델을 사용하여 텍스트 추출
        result = whisper_model.transcribe(tmp_file.name)
        transcript = result["text"]

        # 화자 정보와 텍스트 결합
        output = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end
            output.append({
                "start": start_time,
                "end": end_time,
                "speaker": speaker,
                "text": transcript
            })

    return output

# 문서 ID를 지정하여 처리
document_id = 'your_document_id'
result = process_audio(document_id)
print(result)