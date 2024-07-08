import os
import json
import hashlib
from tts import extract_egs

audio_dir = "audio/"
compressed_dir = "audio_compressed/"

# Get the list of audio files in the audio directory
audio_files = os.listdir(audio_dir)

# Get the list of compressed audio files
compressed_files = os.listdir(compressed_dir)

# Find the missing audio files
missing_files = set(audio_files) - set(compressed_files)

print(f"Missing {len(missing_files)} audio files in {compressed_dir}")

# Load the data from dict.json
with open('../dict.json', 'r') as file:
    data = json.load(file)

egs = extract_egs(data)

# Compare the hash of egs with the missing_files' names
for eg in egs.keys():
    eg_hash = hashlib.sha256(eg.encode()).hexdigest() + ".m4a"
    if eg_hash in missing_files:
        print(f"Missing eg: {eg}, hash: {eg_hash}")

# Delete those missing audios from audio_dir
for eg in egs.keys():
    eg_hash = hashlib.sha256(eg.encode()).hexdigest() + ".m4a"
    if eg_hash in missing_files:
        os.remove(os.path.join(audio_dir, eg_hash))
        print(f"Deleted missing eg: {eg}, hash: {eg_hash}")
