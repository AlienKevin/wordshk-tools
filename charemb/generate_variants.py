import json
import re

chinese_characters_pattern = re.compile(r'^[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U000323af\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007]+$')

with open('../dict.json', 'r') as file:
    data = json.load(file)

def extract_variants(data):
    variant_groups = []
    for entry in data.values():
        variants = entry.get('variants', [])
        variants = sorted(list(set(variant.get('w', '') for variant in variants)))
        if len(variants) > 1 and all(chinese_characters_pattern.match(variant) for variant in variants) \
            and len(set(len(variant) for variant in variants)) == 1 \
            and all(sum(1 if c0 == c1 else 0 for c0, c1 in zip(list(variant), list(variants[0]))) >= len(variants[0]) // 2 for variant in variants):
            variant_groups.append(variants)
    return variant_groups

if __name__ == '__main__':
    variant_groups = extract_variants(data)
    with open('variants.csv', 'w+') as f:
        for variants in variant_groups:
            f.write(' '.join(variants) + '\n')
