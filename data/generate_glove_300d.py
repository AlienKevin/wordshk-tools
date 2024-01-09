import subprocess
import os

# cargo install finalfusion-utils
# google-10000-english.txt from https://github.com/first20hours/google-10000-english/blob/master/google-10000-english.txt
# glove.6B.300d.txt from https://nlp.stanford.edu/data/glove.6B.zip

def load_top_words(file_path):
    with open(file_path, 'r') as file:
        return set(word.strip() for word in file)

def filter_glove_file(glove_path, output_path, top_words):
    num_words = 0
    with open(glove_path, 'r') as glove_file, open(output_path, 'w') as output_file:
        for line in glove_file:
            word = line.split(' ', 1)[0]
            if word in top_words:
                output_file.write(line)
                num_words += 1
    print("Filtered GloVe file to {} words".format(num_words))

def main():
    top_words_file = 'google-10000-english.txt'
    glove_file = 'glove.6B.300d.txt'
    filtered_glove_file = 'filtered_glove.6B.300d.txt'
    finalfusion_file = 'glove.6B.300d.fifu'

    # Load top words and add punctuations and <unk> token
    top_words = load_top_words(top_words_file)
    top_words.update(["<unk>", ".", ",", "!", "?", ";", ":", "'", "\"", "(", ")", "-", "&", "@", "#", "$", "%", "^", "*", "+", "=", "/", "\\"])

    with open('wordshk_english_vocab.txt', 'r') as f:
        for line in f:
            top_words.add(line.strip())

    # Filter GloVe file
    filter_glove_file(glove_file, filtered_glove_file, top_words)

    # Convert to Finalfusion format using finalfusion-utils
    subprocess.run(["finalfusion", "convert", "-f", "text", "-t", "finalfusion", filtered_glove_file, finalfusion_file])

    # Delete the temporary filtered file
    os.remove(filtered_glove_file)

if __name__ == '__main__':
    main()
