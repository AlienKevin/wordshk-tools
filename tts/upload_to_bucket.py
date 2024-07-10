from get_bucket import get_bucket
from tqdm import tqdm
import os
import time

bucket = get_bucket()
def upload_audios():
    uploaded_count = 0
    cutoff_time = time.time() - 1 * 60 * 60  # Only upload files modified within 1 hours ago
    for file in tqdm(os.listdir('audio_compressed')):
        file_path = os.path.join('audio_compressed', file)
        if os.path.getmtime(file_path) > cutoff_time:
            with open(file_path, 'rb') as fileobj:
                bucket.put_object(file, fileobj)
                uploaded_count += 1
    print(f'Number of audios uploaded: {uploaded_count}')

# Run the function to upload .m4a files
upload_audios()
