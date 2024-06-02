from transformers import BertModel, BertTokenizer
import torch
from dataset import get_dataset
from cluster import plot_embeddings
from tqdm import tqdm

# Load the BERT model and tokenizer
model_name = 'indiejoseph/bert-base-cantonese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to extract BERT embeddings
def extract_bert_embeddings(text, model, tokenizer, device='cpu'):
    model.to(device)
    model.eval()
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings from the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings

if __name__ == '__main__':
    sample_text = "你好，世界！"  # Example text in Cantonese
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dataloader, num_char_classes, num_jyutping_classes = get_dataset()

    all_embeddings = {}
    for inputs, characters, char_labels, jyutping_labels in tqdm(dataloader):
        for i, text in enumerate(characters):
            embeddings = extract_bert_embeddings(text, model, tokenizer, device)
            all_embeddings[text] = embeddings.cpu()

    torch.save(all_embeddings, 'bert_vecs.pt')

    labels = list(all_embeddings.keys())
    embeddings = torch.cat(list(all_embeddings.values())).numpy()

    plot_embeddings(embeddings, labels)
