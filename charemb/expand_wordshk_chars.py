import opencc
import re
import json

cjk_chars_pattern = '\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U000323af\U0002ebf0-\U0002ee5f\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3007'

cjk_regex = re.compile(
    '['
    f'{cjk_chars_pattern}'
    ']'
)

# Initialize the OpenCC converter
converter = opencc.OpenCC('t2s')

# Read traditional characters from wordshk_chars.txt
with open('wordshk_chars.txt', 'r', encoding='utf-8') as file:
    traditional_chars = file.read().splitlines()
    print(f'traditional_chars: {len(traditional_chars)}')

# Convert traditional characters to simplified characters
simplified_chars = [converter.convert(char) for char in traditional_chars if char != converter.convert(char)]
print(f'simplified_chars: {len(simplified_chars)}')

# Read CASIA characters from casia_charset.txt
with open('casia_charset.txt', 'r', encoding='utf-8') as file:
    casia_charset_chars = file.read().splitlines()
    print(f'casia_chars: {len(casia_charset_chars)}')

# Read HKSCS2016.json and extract the Unicode chars that are Chinese characters that satisfy chinese_characters_pattern
with open('HKSCS2016.json', 'r', encoding='utf-8-sig') as file:
    hkscs_data = json.load(file)
    hkscs_chars = [entry['char'] for entry in hkscs_data if cjk_regex.match(entry['char'])]
    print(f'hkscs_chars: {len(hkscs_chars)}')

# Read big5-2003.txt and extract the right column as chars
with open('big5-2003.txt', 'r', encoding='utf-8') as file:
    big5_2003_chars = []
    for line in file:
        parts = line.split()
        assert len(parts) == 2, f'Found an invalid line in big5-2003.txt: {line}'
        char = chr(int(parts[1], 16))
        if cjk_regex.match(char):
            big5_2003_chars.append(char)

print(f'big5_2003_chars: {len(big5_2003_chars)}')

# Union the original traditional characters, simplified characters, and CASIA characters
expanded_chars = set(traditional_chars + simplified_chars + casia_charset_chars + hkscs_chars + big5_2003_chars)

print(f'expanded_chars: {len(expanded_chars)}')

# Write the expanded characters to wordshk_chars_expanded.txt
with open('wordshk_chars_expanded.txt', 'w', encoding='utf-8') as file:
    for char in sorted(expanded_chars):
        file.write(char + '\n')
