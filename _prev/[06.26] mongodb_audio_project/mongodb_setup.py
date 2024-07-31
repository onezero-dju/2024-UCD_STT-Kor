from pymongo import MongoClient

# MongoDB에 연결
client = MongoClient('localhost', 27017)

# 데이터베이스 및 컬렉션 설정
db = client['voice_data_db']    # voice_data_db: DB명
collection = db['pcm_files']    # collection: 문서의 그룹

print("MongoDB 연결 및 데이터베이스 설정 완료")