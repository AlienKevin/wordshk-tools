import os
from datasets import Dataset, Audio
from huggingface_hub import HfApi, HfFolder
import json
from tts import extract_defs_and_egs, normalize, normalize_file_name

def collect_audios_to_dataset(sents, audio_folder):
    data = {"audio": [], "text": []}

    normalized_sents = {normalize_file_name(normalize(sent)): sent for sent in sents.keys()}
    
    for root, _, files in os.walk(audio_folder):
        for file in files:
            if file.endswith(".mp3"):
                file_path = os.path.join(root, file)
                text = os.path.splitext(file)[0]  # Assuming the text is the filename without extension
                if text in normalized_sents:
                    sent = normalized_sents[text]
                    if sent != text:
                        print(f"Before normalize: {sent}")
                        print(f"After normalize: {text}")
                    text = sent
                else:
                    print(f"Cannot find a matching sentence for the audio file: {file}")
                data["audio"].append(file_path)
                data["text"].append(text)
    
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio())
    return dataset

def upload_audio_to_hf(dataset):
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("You need to login to the Hugging Face Hub first. Run `huggingface-cli login`.")
    repo_id = "AlienKevin/wordshk_cantonese_speech"
    dataset.push_to_hub(repo_id, token=token)

if __name__ == "__main__":
    audio_folder = "audio"

     # Load the data from dict.json
    with open('../dict.json', 'r') as file:
        data = json.load(file)
    
    sents = extract_defs_and_egs(data)

    dataset = collect_audios_to_dataset(sents, audio_folder)

    upload_audio_to_hf(dataset)
