from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import json
from threading import Lock
import random
import pycantonese
import re

random.seed(42)

with open('deepseek_api_key.txt', 'r') as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

valid_pos_tags = {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"}

def segment_words(sentence):
    attempts = 0
    while True:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{
                            "role": "system",
                            "content": pos_prompt
                          },
                          {
                            "role": "user",
                            "content": sentence
                          }],
                response_format={
                    'type': 'json_object'
                },
                max_tokens=2048,
                temperature=1.1,
                stream=False
            )
            result = response.choices[0].message.content
            try:
                result_json = json.loads(result)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse JSON response: {str(e)}")
            if "pos_tagged_words" in result_json and isinstance(result_json["pos_tagged_words"], list) and \
                all(isinstance(item, list) and len(item) == 2 and (all(isinstance(sub_item, str) for sub_item in item)) for item in result_json["pos_tagged_words"]):
                concatenated_words = "".join([word for word, pos in result_json["pos_tagged_words"]])
                if concatenated_words == sentence:
                    for word, pos in result_json["pos_tagged_words"]:
                        if pos not in valid_pos_tags:
                            raise Exception(f"Invalid POS tag '{pos}' in the result")
                    return result_json["pos_tagged_words"]
                else:
                    raise Exception(f"Segmentation result does not match the input sentence")
            else:
                raise Exception(f"Invalid segmentation result format")
        except Exception as e:
            time.sleep(1)
            attempts += 1
            if attempts >= 3:
                return {'error': str(e), 'result': result}


def generate_prompt():
    # Load HKCanCor data
    hkcancor_data = pycantonese.hkcancor()

    # Gather all word segmented utterances
    utterances = [[(token.word, pycantonese.pos_tagging.hkcancor_to_ud(token.pos)) for token in utterance] for utterance in hkcancor_data.tokens(by_utterances=True)]

    latin_fragmented_utterances = []
    latin_complete_utterances = []
    non_latin_utterances = []

    for utterance in utterances:
        if any(re.search(r'[a-zA-Z0-9]', char) for (word, pos) in utterance for char in word):
            if ''.join(word for word, pos in utterance).count('"') % 2 != 0:
                latin_fragmented_utterances.append(utterance)
            else:
                latin_complete_utterances.append(utterance)
        else:
            non_latin_utterances.append(utterance)

    # Sample 3 random latin_complete_utterances, 3 random latin_fragmented_utterances, and 4 random non_latin_utterances for in_context_samples
    random.shuffle(latin_complete_utterances)
    random.shuffle(latin_fragmented_utterances)
    random.shuffle(non_latin_utterances)
    in_context_samples = latin_complete_utterances[:3] + latin_fragmented_utterances[:3] + non_latin_utterances[:4]
    random.shuffle(in_context_samples)
    testing_samples = latin_complete_utterances[3:] + latin_fragmented_utterances[3:] + non_latin_utterances[4:]

    # Format in-context samples for the prompt
    in_context_prompt = "\n\n".join([
        f'EXAMPLE INPUT SENTENCE:\n{"".join([word for word, pos in sample])}\n\nEXAMPLE JSON OUTPUT:\n{json.dumps({"pos_tagged_words": [[word, pos] for word, pos in sample]}, ensure_ascii=False)}'
        for sample in in_context_samples
    ])

    # Update the word segmentation prompt with in-context samples
    pos_prompt = f"""You are an expert at Cantonese word segmentation and POS tagging. Output the segmented words with their parts of speech as JSON arrays. ALWAYS preserve typos, fragmented chunks, and punctuations in the original sentence. Output the parts of speech in the Universal Dependencies v2 tagset with 17 tags:
ADJ: adjective
ADP: adposition
ADV: adverb
AUX: auxiliary
CCONJ: coordinating conjunction
DET: determiner
INTJ: interjection
NOUN: noun
NUM: numeral
PART: particle
PRON: pronoun
PROPN: proper noun
PUNCT: punctuation
SCONJ: subordinating conjunction
SYM: symbol
VERB: verb
X: other

{in_context_prompt}"""

    # Write the updated word segmentation prompt to the file
    with open('pos_prompt.txt', 'w', encoding='utf-8') as f:
        f.write(pos_prompt)
    
    return pos_prompt, testing_samples

if __name__ == "__main__":
    pos_prompt, testing_samples = generate_prompt()
    with open('hkcancor_pos_tagged.jsonl', 'w', encoding='utf-8') as file, open('hkcancor_pos_tagged_errors.jsonl', 'w', encoding='utf-8') as error_file:
        lock = Lock()
        def process_sample(sample):
            input_sentence = "".join([word for word, pos in sample])
            pos_result = segment_words(input_sentence)
            if 'error' in pos_result:
                print(f"POS tagging failed for sentence: {input_sentence}")
                print(f"Error: {pos_result['error']}")
                error_result = {
                    "reference": sample,
                    "error": pos_result['error'],
                    "hypothesis": pos_result['result']
                }
                with lock:
                    error_file.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                    error_file.flush()
                result = {
                    "reference": sample,
                    "hypothesis": [(char, "X") for char in list(sample)]
                }
            else:
                result = {
                    "reference": sample,
                    "hypothesis": pos_result
                }
            with lock:
                file.write(json.dumps(result, ensure_ascii=False) + '\n')
                file.flush()
        with ThreadPoolExecutor(max_workers=100) as executor:
            list(tqdm(executor.map(process_sample, testing_samples), total=len(testing_samples)))
