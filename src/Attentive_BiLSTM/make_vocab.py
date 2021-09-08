from collections import Counter


def read_file(ip_file, vocab):
    for line in open(ip_file):
        # print(line)
        if line.strip() == "":
            continue
        line_values = line.strip().split(' ')
        word = line_values[0]

        vocab[word] += 1


if __name__ == '__main__':
    vocab = Counter()

    ip_file = "pred.dev_7"
    read_file(ip_file, vocab)

    ip_file = "pred.test_7"
    read_file(ip_file, vocab)

    ip_file = "pred.train_7"
    read_file(ip_file, vocab)

    with open("vocab.tsv", 'w') as fout:
        for w in vocab:
            print(w, vocab[w])
            opline = w + "\t" + str(vocab[w]) + "\n"
            fout.write(opline)
