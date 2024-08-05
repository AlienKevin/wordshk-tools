import json

def process_entries(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            entry = json.loads(line)
            # Remove the specified fields
            entry.pop('variants', None)
            entry.pop('yueDef', None)
            entry.pop('engDef', None)
            entry.pop('egs', None)
            # Write the modified entry to the output file
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Specify the input and output file paths
input_file_path = 'outputs/entry_results.jsonl'
output_file_path = 'outputs/mandarin_entries.jsonl'

# Process the entries
process_entries(input_file_path, output_file_path)
