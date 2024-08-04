import json

# Read the error data
with open('hkcancor_pos_tagged_errors.jsonl', 'r', encoding='utf-8') as error_file:
    error_data = [json.loads(line) for line in error_file]

import difflib

red = lambda text: f"\033[38;2;255;0;0m{text}\033[38;2;255;255;255m"
green = lambda text: f"\033[38;2;0;255;0m{text}\033[38;2;255;255;255m"
blue = lambda text: f"\033[38;2;0;0;255m{text}\033[38;2;255;255;255m"
white = lambda text: f"\033[38;2;255;255;255m{text}\033[38;2;255;255;255m"

def get_edits_string(old, new):
    result = ""
    codes = difflib.SequenceMatcher(a=old, b=new).get_opcodes()
    for code in codes:
        if code[0] == "equal": 
            result += white(old[code[1]:code[2]])
        elif code[0] == "delete":
            result += red(old[code[1]:code[2]])
        elif code[0] == "insert":
            result += green(new[code[3]:code[4]])
        elif code[0] == "replace":
            result += (red(old[code[1]:code[2]]) + green(new[code[3]:code[4]]))
    return result

# Visualize the error between the reference and hypothesis
for entry in error_data:
    # Filter out entries with the specific error
    if entry['error'] != "Segmentation result does not match the input sentence":
        print('ERROR:', entry['error'])
        print(entry)
        continue

    reference_tokens = [token for token, pos in entry['reference']]
    hypothesis_tokens = [token for token, pos in json.loads(entry['hypothesis'])['pos_tagged_words']]
    
    reference_sentence = "".join(reference_tokens)
    hypothesis_sentence = "".join(hypothesis_tokens)
    
    diff = get_edits_string(reference_sentence, hypothesis_sentence)
    
    print(diff)
