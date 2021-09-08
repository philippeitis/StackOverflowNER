import json
from pathlib import Path

import numpy as np


def read_file(path, dest: Path = Path("./word_to_id.json")):
    all_freq_embed = {}
    word_to_id = {}
    current_word_id = 0
    for line in open(path):
        s = line.strip().split()
        word = s[0]
        word_to_id[word] = current_word_id
        current_word_id += 1

        all_freq_embed[s[0]] = np.array([float(i) for i in s[1:]])

    word_to_id["UNK"] = current_word_id
    current_word_id += 1
    word_to_id["***PADDING***"] = current_word_id

    dest.write_text(json.dumps(word_to_id))

    print(len(word_to_id))
    freq_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), 102))
    for w in word_to_id:
        if w in all_freq_embed:
            freq_embeds[word_to_id[w]] = all_freq_embed[w]
        elif w.lower() in all_freq_embed:
            freq_embeds[word_to_id[w]] = all_freq_embed[w.lower()]

    np.save('freq_embeds.npy', freq_embeds)


def main():
    path = "Freq_Vector.txt"
    read_file(path)
    word_to_id = json.load(open("word_to_id.json", "r"))
    print(len(word_to_id))


if __name__ == '__main__':
    main()
