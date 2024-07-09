import oss2
from get_bucket import get_bucket
from tqdm import tqdm
import os

bucket = get_bucket()

def upload_audios():
    for file in tqdm(os.listdir('audio_compressed')):
        file_path = os.path.join('audio_compressed', file)
        with open(file_path, 'rb') as fileobj:
            bucket.put_object(file, fileobj)

# Run the function to upload .m4a files
upload_audios()
