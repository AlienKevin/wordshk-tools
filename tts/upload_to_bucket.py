from get_bucket import get_bucket
from tqdm import tqdm
import os
import time

bucket = get_bucket()
def upload_audios():
    uploaded_count = 0
    reupload_queue = []
    cutoff_time = time.time() - 100 * 60 * 60  # Only upload files modified within 1 hours ago
    for file in tqdm(os.listdir('audio')):
        file_path = os.path.join('audio', file)
        if os.path.getmtime(file_path) > cutoff_time and file.endswith('.mp3'):
            with open(file_path, 'rb') as fileobj:
                try:
                    bucket.put_object(file, fileobj)
                    uploaded_count += 1
                except Exception as e:
                    print(f"Failed to upload {file}: {e}")
                    reupload_queue.append(file_path)
    
    # Attempt to reupload files in the reupload queue
    for file_path in reupload_queue:
        with open(file_path, 'rb') as fileobj:
            try:
                bucket.put_object(os.path.basename(file_path), fileobj)
                uploaded_count += 1
                reupload_queue.remove(file_path)
            except Exception as e:
                print(f"Failed to reupload {file_path}: {e}")

    print(f'Number of audios uploaded: {uploaded_count}')
    if reupload_queue:
        print(f'Files that failed to upload: {reupload_queue}')

# Run the function to upload .m4a files
upload_audios()
