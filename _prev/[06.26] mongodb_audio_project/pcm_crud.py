from bson import ObjectId
from mongodb_setup import db, collection

# Create (생성): 로컬에 있는 PCM 파일을 MongoDB에 저장 
def create_pcm_file(file_path, metadata):
    with open(file_path, 'rb') as file:
        file_data = file.read()
    document = {
        'filename': file_path,
        'file_data': file_data,
        'metadata': metadata
    }
    result = collection.insert_one(document)
    print(f'File inserted with id: {result.inserted_id}')

# Read (읽기): MongoDB에서 PCM 파일을 읽고 로컬에 저장
def read_pcm_file(file_id, save_path):
    document = collection.find_one({'_id': file_id})
    if document:
        with open(save_path, 'wb') as file:
            file.write(document['file_data'])
        print(f'File saved to: {save_path}')
    else:
        print('File not found')

# Update (업데이트): MongoDB에 저장된 PCM 파일의 메타데이터를 업데이트
def update_pcm_metadata(file_id, new_metadata):
    result = collection.update_one({'_id': file_id}, {'$set': {'metadata': new_metadata}})
    if result.modified_count > 0:
        print('Metadata updated successfully')
    else:
        print('No document found with the given id')

# Delete (삭제): MongoDB에서 PCM 파일을 삭제
def delete_pcm_file(file_id):
    result = collection.delete_one({'_id': file_id})
    if result.deleted_count > 0:
        print('File deleted successfully')
    else:
        print('No document found with the given id')
