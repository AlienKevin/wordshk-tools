from transformers import WhisperProcessor, WhisperModel
import torch
from dataset import get_dataset
from cluster import plot_embeddings
from tqdm import tqdm

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

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    from pathlib import Path
    from pydub import AudioSegment

    audio_dir = Path('jyutping_female')
    all_embeddings = {}

    for audio_file in tqdm(list(audio_dir.glob('*.mp3'))):
        jyutping_syllable = audio_file.stem
        audio = AudioSegment.from_mp3(audio_file).get_array_of_samples()
        embeddings = extract_whisper_embeddings(audio, model, processor, device)
        all_embeddings[jyutping_syllable] = embeddings.cpu()

    torch.save(all_embeddings, 'whisper_vecs.pt')

    labels = list(all_embeddings.keys())
    embeddings = torch.cat(list(all_embeddings.values())).numpy()

    plot_embeddings(embeddings, labels)
