import os
import json
import hashlib
from tts import extract_egs

audio_dir = "audio/"

# Load the data from dict.json
with open('../dict.json', 'r') as file:
    data = json.load(file)

egs = extract_egs(data)

# Verify all egs are generated in audio/

# Get the list of audio files in the audio directory
audio_files = os.listdir(audio_dir)

# Find the missing egs
missing_egs = []
for eg in egs.keys():
    eg_hash = hashlib.sha256(eg.encode()).hexdigest() + ".m4a"
    if eg_hash not in audio_files:
        missing_egs.append(eg)

print(f"Missing {len(missing_egs)} egs in {audio_dir}")
