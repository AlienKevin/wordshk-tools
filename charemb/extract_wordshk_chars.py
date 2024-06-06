import json
import re

chinese_characters_pattern = re.compile(r'^[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U000323af\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007]+$')

with open('../dict.json', 'r') as file:
    data = json.load(file)

def extract_chars(data):
    chars = set()
    for entry in data.values():
        variants = entry.get('variants', [])
        variants = set(variant.get('w', '') for variant in variants)
        for variant in variants:
            chars.update(set(variant))
    return chars

if __name__ == '__main__':
    chars = extract_chars(data)
    with open('wordshk_chars.txt', 'w+') as f:
        for c in chars:
            if chinese_characters_pattern.match(c):
                f.write(c + '\n')
