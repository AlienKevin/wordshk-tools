import opencc

# Initialize the OpenCC converter
converter = opencc.OpenCC('t2s')

# Read traditional characters from wordshk_chars.txt
with open('wordshk_chars.txt', 'r', encoding='utf-8') as file:
    traditional_chars = file.read().splitlines()

# # Convert traditional characters to simplified characters
# simplified_chars = [converter.convert(char) for char in traditional_chars]
simplified_chars = []

# Read CASIA characters from casia_charset.txt
with open('casia_charset.txt', 'r', encoding='utf-8') as file:
    casia_charset_chars = file.read().splitlines()

# Union the original traditional characters, simplified characters, and CASIA characters
expanded_chars = set(traditional_chars + simplified_chars + casia_charset_chars)

# Write the expanded characters to wordshk_chars_expanded.txt
with open('wordshk_chars_expanded.txt', 'w', encoding='utf-8') as file:
    for char in sorted(expanded_chars):
        file.write(char + '\n')
