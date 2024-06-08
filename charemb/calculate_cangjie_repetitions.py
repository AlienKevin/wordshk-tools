import matplotlib.pyplot as plt
from collections import defaultdict

def count_cangjie_repetitions(file_path, included_chars_file):
    cangjie_dict = defaultdict(list)
    
    # Load valid characters from wordshk_chars_expanded.txt
    with open(included_chars_file, 'r', encoding='utf-8') as f:
        valid_chars = set(line.strip() for line in f)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                character = parts[0]
                if character in valid_chars:
                    cangjie_codes = parts[1].split()
                    for code in cangjie_codes:
                        cangjie_dict[code].append(character)
    
    repetition_counts = defaultdict(int)
    for characters in cangjie_dict.values():
        repetition_counts[len(characters)] += 1
    
    return repetition_counts
def plot_repetition_counts(repetition_counts):
    counts = sorted(repetition_counts.items())
    x, y = zip(*counts)
    plt.bar(x, y, width=0.8)  # Enlarge bar width
    plt.xlabel('Number of Repeats')
    plt.ylabel('Frequency')
    plt.title('Cangjie Code Repetition Counts')
    # Add counts on top of bars
    for i, count in enumerate(y):
        plt.text(x[i], count, str(count), ha='center', va='bottom')
    # Show x labels under each bar
    plt.xticks(range(len(x)), x)
    # Stop at max count
    plt.ylim(0, max(y) + 1)
    plt.show()

if __name__ == '__main__':
    file_path = 'Cangjie5_HK.txt'
    included_chars_file = 'wordshk_chars_expanded.txt'
    repetition_counts = count_cangjie_repetitions(file_path, included_chars_file)
    print(sorted(list(repetition_counts.items())))
    plot_repetition_counts(repetition_counts)
