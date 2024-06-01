import os
import subprocess

audio_compressed_dir = 'audio_compressed'
if not os.path.exists(audio_compressed_dir):
    os.makedirs(audio_compressed_dir)

import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for filename in os.listdir('audio'):
        if filename.endswith(".m4a"):
            input_file_path = os.path.join('audio', filename)
            output_file_path = os.path.join(audio_compressed_dir, filename)
            command = f'ffmpeg -i {input_file_path} -c:a libfdk_aac -vbr 1 {output_file_path}'
            futures.append(executor.submit(lambda cmd: subprocess.run(cmd, shell=True), command))
    concurrent.futures.wait(futures)
