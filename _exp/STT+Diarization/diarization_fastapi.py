from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisper
from pyannote.audio import Pipeline
from pymongo import MongoClient
import tempfile
import wave

app = FastAPI()

# MongoDB 설정
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database']
collection = db['your_collection']

# 모델 로드
whisper_model = whisper.load_model("base")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                use_auth_token="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")

class ProcessRequest(BaseModel):
    document_id: str

class SpeakerSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str

@app.post("/process-audio", response_model=list[SpeakerSegment])
def process_audio(request: ProcessRequest):
    document = collection.find_one({'_id': request.document_id})
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    audio_data = document.get('audio_data')
    if not audio_data:
        raise HTTPException(status_code=400, detail="No audio data found")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        with wave.open(tmp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)
        tmp_file.flush()

        # 화자 다이어리제이션
        diarization = diarization_pipeline(tmp_file.name)
        
        # STT
        result = whisper_model.transcribe(tmp_file.name)
        transcript = result["text"]

        # 결과 결합
        output = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end
            output.append(SpeakerSegment(
                start=start_time,
                end=end_time,
                speaker=speaker,
                text=transcript
            ))

    return output
