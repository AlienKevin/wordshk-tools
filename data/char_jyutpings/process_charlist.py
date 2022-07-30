import json

with open('charlist.json', 'r') as input_file:
    with open('charlist_processed.json', 'w+') as output_file:
        content = json.dumps(json.loads(input_file.read()), ensure_ascii=False,
                             indent=None, separators=(',', ':'))
        output_file.write(content)
