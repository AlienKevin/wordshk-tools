import pycantonese
import json
import random

# Patches https://github.com/jacksonllee/pycantonese/issues/48
def patch_pycantonese_tag_bug(tag):
    if tag == "V":
        return "VERB"
    else:
        return tag


def load_hkcancor():
    # Load HKCanCor data
    hkcancor_data = pycantonese.hkcancor()

    # Gather all word segmented utterances
    utterances = [[(token.word, patch_pycantonese_tag_bug(pycantonese.pos_tagging.hkcancor_to_ud(token.pos))) for token in utterance] for utterance in hkcancor_data.tokens(by_utterances=True)]

    return utterances


system_prompt = """You are an expert at Cantonese word segmentation and POS tagging. Output the parts of speech in the Universal Dependencies v2 tagset with 17 tags in the JSON format.

EXAMPLE INPUT SENTENCE:
即係大陸-,香港就係office來嘅.

EXAMPLE JSON OUTPUT:
{"pos_tagged_words": [["即係", "CCONJ"], ["大陸", "PROPN"], ["-", "PUNCT"], [",", "PUNCT"], ["香港", "PROPN"], ["就", "ADV"], ["係", "VERB"], ["office", "NOUN"], ["來", "PART"], ["嘅", "PART"], [".", "PUNCT"]]}"""


def create_sft(utterances, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for utterance in utterances:
            sentence = "".join([word for word, _ in utterance])
            expected_output =json.dumps({"pos_tagged_words": [[word, pos] for word, pos in utterance]}, ensure_ascii=False)
            sft_entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sentence},
                    {"role": "assistant", "content": expected_output}
                ]
            }
            f.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")


def create_sft_jsonl_dataset(utterances, output_file, train_size=2600, validation_size=1000):
    random.seed(42)

    random.shuffle(utterances)

    validation_dataset = utterances[:validation_size]
    train_dataset = utterances[validation_size:validation_size+train_size]

    create_sft(validation_dataset, 'data/hkcancor_sft_validation.jsonl')
    create_sft(train_dataset, 'data/hkcancor_sft_train.jsonl')
    
if __name__ == "__main__":
    # Example usage
    utterances = load_hkcancor()
    create_sft_jsonl_dataset(utterances, 'data/hkcancor_sft_dataset.jsonl')
