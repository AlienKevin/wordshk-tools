import json
import re

chinese_characters_pattern = re.compile(r'^[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U000323af\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007]+$')

with open('../dict.json', 'r') as file:
    data = json.load(file)

def extract_chars(data):
    chars = set()

    def extract_text(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                yield from extract_text(value)
        elif isinstance(obj, list):
            for item in obj:
                yield from extract_text(item)
        elif isinstance(obj, str):
            yield obj

    for text in extract_text(data):
        chars.update(text)

    with open('../data/char_jyutpings/charlist.json', 'r') as file:
        charlist_data = json.load(file)
        chars.update(charlist_data.keys())
    
    with open('../data/hk_variant_map_safe.tsv', 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            assert len(columns) == 2
            chars.update(columns[0])
            chars.update(columns[1])

    return sorted(list(chars))

if __name__ == '__main__':
    chars = extract_chars(data)
    with open('wordshk_chars.txt', 'w+') as f:
        for c in chars:
            if chinese_characters_pattern.match(c):
                f.write(c + '\n')
