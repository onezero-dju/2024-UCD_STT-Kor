from bson import ObjectId
from pcm_crud import create_pcm_file, read_pcm_file, update_pcm_metadata, delete_pcm_file
from mongodb_setup import collection

def main():
    # 절대 경로 설정 (예시 경로입니다. 실제 경로로 변경하세요)
    abs_file_path = "C:\\Users\\yyt11\\OneDrive\\바탕 화면\\mongodb_audio_project\\Korean Voice\\KsponSpeech_01\\KsponSpeech_0001\\KsponSpeech_000001.pcm"
    abs_save_path = "C:\\Users\\yyt11\\OneDrive\\바탕 화면\\mongodb_audio_project\\Korean Voice repo\\KsponSpeech_01\\KsponSpeech_0001\\retrieved_audio.pcm"

    # 1. PCM 파일 생성
    print("Creating PCM file...")
    create_pcm_file(abs_file_path, {'description': 'Sample PCM file'})

    # 2. 방금 생성한 파일의 ObjectId 가져오기
    created_file = collection.find_one({'filename': abs_file_path})
    file_id = created_file['_id']
    print(f"Created file with ID: {file_id}")

    # 3. PCM 파일 읽기
    print("Reading PCM file...")
    read_pcm_file(file_id, abs_save_path)

    # 4. 메타데이터 업데이트
    print("Updating PCM metadata...")
    update_pcm_metadata(file_id, {'description': 'Updated description'})

    # 5. PCM 파일 삭제
    print("Deleting PCM file...")
    delete_pcm_file(file_id)
    print(f"Deleted file with ID: {file_id}")

if __name__ == "__main__":
    main()
