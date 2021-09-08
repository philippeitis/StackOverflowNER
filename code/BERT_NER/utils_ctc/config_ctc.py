from pathlib import Path

BASE_CTC_DIR = Path("/home/philippe/PycharmProjects/StackOverflowNER/data_ctc")
FASTTEXT_BIN = "/home/philippe/PycharmProjects/StackOverflowNER/resources/fasttext.bin"

parameters_ctc = {
    'train_file': str(BASE_CTC_DIR / Path("train.tsv")),
    'test_file': str(BASE_CTC_DIR / Path("test.tsv")),
    'LR': 0.0015,
    'epochs': 70,
    'word_dim': 300,
    'hidden_layer_1_dim': 300,
    "RESOURCES_base_directory": BASE_CTC_DIR,
}
