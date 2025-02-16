import json
import random

def add_character_noise(word):
    if len(word) > 1:
        noise_type = random.choice(['swap', 'delete', 'insert'])
        idx = random.randint(0, len(word) - 2) if noise_type == 'swap' else random.randint(0, len(word) - 1)
        random_char = random.choice('abcdefghijklmnopqrstuvwxyz')

        if noise_type == 'swap':
            word = word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
        elif noise_type == 'delete':
            word = word[:idx] + word[idx+1:]
        elif noise_type == 'insert':
            word = word[:idx] + random_char + word[idx:]
    return word

def add_word_noise(word, words):
    noise_type = random.choice(['delete', 'replace', 'swap'])
    if noise_type == 'delete':
        return None
    elif noise_type == 'replace':
        return random.choice(['the', 'a', 'and', 'is', 'in'])
    elif noise_type == 'swap' and len(words) > 1:
        idx = random.randint(0, len(words) - 1)
        words[idx], word = word, words[idx]
    return word

def add_punctuation_noise(word):
    if word.isalpha():
        word = word + random.choice(['.', ',', '!', '?'])
    return word

def apply_noise_to_word(word, words, char_prob=0.1, word_prob=0.1, punct_prob=0.1, char_noise=False, word_noise=False, punct_noise=False):
    if char_noise and random.random() < char_prob:
        word = add_character_noise(word)

    if word_noise and random.random() < word_prob:
        word = add_word_noise(word, words) or word

    if punct_noise and random.random() < punct_prob:
        word = add_punctuation_noise(word)

    return word

label_mapping = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

def add_noise_to_snli(input_path, output_path, char_prob=0.3, word_prob=0.3, punct_prob=0.1, char_noise=False, word_noise=False, punct_noise=False):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            try:
                example = json.loads(line)

                if example['gold_label'] == '-':
                    continue
                
                # Apply noise to sentence1 (premise)
                words = example['sentence1'].split()
                noisy_words = []
                for word in words:
                    noisy_word = apply_noise_to_word(word, words, char_prob, word_prob, punct_prob, char_noise, word_noise, punct_noise)
                    if noisy_word is not None:
                        noisy_words.append(noisy_word)
                example['sentence1'] = ' '.join(noisy_words)

                # Apply noise to sentence2 (Hypothesis)
                words = example['sentence2'].split()
                noisy_words = []
                for word in words:
                    noisy_word = apply_noise_to_word(word, words, char_prob, word_prob, punct_prob, char_noise, word_noise, punct_noise)
                    if noisy_word is not None:
                        noisy_words.append(noisy_word)
                example['sentence2'] = ' '.join(noisy_words)
                
                filtered_example = {
                    "premise": example.pop('sentence1'),
                    "hypothesis": example.pop('sentence2'),
                    "label": label_mapping.get(example.pop('gold_label', None), -1)
                }
                
                # Write the filtered example to the new JSONL file
                outfile.write(json.dumps(filtered_example) + '\n')
            except json.JSONDecodeError:
                print("Skipping malformed line")
    print("Processing done")

if __name__ == "__main__":
    input_file = 'snli_1.0\\snli_1.0_test.jsonl'
    output_file = 'noisy_premise_c.jsonl'
    add_noise_to_snli(input_file, output_file, char_prob=1, word_prob=1, punct_prob=0.2, char_noise=True, word_noise=True, punct_noise=False)
