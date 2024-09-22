import whisper
from pyannote.audio import Pipeline
import numpy as np
import tempfile
import wave
import matplotlib.pyplot as plt

# Whisper 모델과 pyannote.audio 파이프라인 불러오기
whisper_model = whisper.load_model("base")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")

# 화자 분리 및 STT 수행
def process_audio(file_path):
    # 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        with wave.open(file_path, 'rb') as wf:
            params = wf.getparams()
            audio_data = wf.readframes(params.nframes)
        with wave.open(tmp_file, 'wb') as wf:
            wf.setparams(params)
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

    return output, diarization, transcript

# 결과 시각화
def visualize_diarization(diarization, transcript):
    fig, ax = plt.subplots()

    # 화자 분리 결과 표시
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        ax.fill_betweenx([speaker], turn.start, turn.end, alpha=0.5)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speaker")
    ax.set_title("Speaker Diarization")
    plt.show()

    # 텍스트 결과 출력
    print("Transcript:")
    print(transcript)

# 로컬 파일 경로를 지정하여 처리
file_path = './Users/minhyeok/Desktop/PROJECT/STTtest_file/whisper_stt/mhspeak.mp3'
result, diarization, transcript = process_audio(file_path)
visualize_diarization(diarization, transcript)