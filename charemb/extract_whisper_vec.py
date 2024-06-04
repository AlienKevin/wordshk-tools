from transformers import WhisperProcessor, WhisperModel
import torch
from cluster import plot_embeddings
from tqdm import tqdm
import json
from utils import normalize

# Load the Whisper model and processor
model_name = 'alvanlii/whisper-small-cantonese'
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperModel.from_pretrained(model_name)

# Function to extract Whisper embeddings
def extract_whisper_embeddings(audio, model, processor, device='cpu'):
    model.to(device)
    model.eval()
    
    # Process the input audio
    inputs = processor(audio, return_tensors='pt', sampling_rate=16000)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    
    # Extract the embeddings from the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings

with open('../data/char_jyutpings/charlist_processed.json', 'r') as f:
    char_jyutpings = json.load(f)

def get_jyutpings(char):
    return char_jyutpings.get(char, [])


if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    from pathlib import Path
    from pydub import AudioSegment

    audio_dir = Path('jyutping_female')
    vecs = {}

    for audio_file in tqdm(sorted(audio_dir.glob('*.mp3'))):
        jyutping_syllable = audio_file.stem
        audio = AudioSegment.from_mp3(audio_file).get_array_of_samples()
        embeddings = extract_whisper_embeddings(audio, model, processor, device)
        vecs[jyutping_syllable] = embeddings.cpu()

    torch.save(vecs, 'whisper_vecs.pt')
    labels = list(vecs.keys())
    embeddings = torch.cat(list(vecs.values())).numpy()
    plot_embeddings(embeddings, labels)

    char_vecs = {}
    for char in sorted(char_jyutpings.keys()):
        char_vec = [torch.zeros_like(next(iter(vecs.values())))] # have at least one element for torch.stack
        jyutpings = get_jyutpings(char)
        for jyutping in jyutpings:
            if jyutping in vecs:
                char_vec.append(normalize(vecs[jyutping]))
        char_vecs[char] = torch.mean(torch.stack(char_vec), dim=0)
    
    torch.save(char_vecs, 'whisper_char_vecs.pt')
    labels = list(char_vecs.keys())
    embeddings = torch.cat(list(char_vecs.values())).numpy()
    plot_embeddings(embeddings, labels)
