import os
import subprocess
from tqdm import tqdm

audio_compressed_dir = 'audio_compressed'
if not os.path.exists(audio_compressed_dir):
    os.makedirs(audio_compressed_dir)

import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for filename in os.listdir('audio'):
        if filename.endswith(".m4a"):
            input_file_path = os.path.join('audio', filename)
            output_file_path = os.path.join(audio_compressed_dir, os.path.splitext(filename)[0] + '.mp3')
            if not os.path.exists(output_file_path):
                command = f'ffmpeg -i {input_file_path} -codec:a libmp3lame -qscale:a 2 {output_file_path}'
                futures.append(executor.submit(lambda cmd: subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), command))
    
    for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Compressing audio files"):
        pass

    print(f"Number of generated compressed files: {len(futures)}")
