from .tokenizer_interface import Tokenizer

# vocab_size = 256 (base bytes) + max_merges
# e.g. max_merges=744 → vocab_size=1000, matching CONFIG['vocab_size']
MAX_MERGES = 744

class BPETokenizer(Tokenizer):
    def __init__(self, max_merges=MAX_MERGES):
        self.max_merges = max_merges
        self.pairs_dict = {}
        self.vocab = {}

    def ingest_docs(self, file_path: str):
        file_content = self.read_file(file_path)
        self.train(file_content)

    def train(self, text: str):
        self.pairs_dict = {}
        self.vocab = {}

        file_ids = list(text.encode('utf-8'))
        for i in range(self.max_merges):
            pairs = self.create_pairs(file_ids)
            if not pairs:
                break
            max_pair = max(pairs, key=pairs.get)
            self.pairs_dict[max_pair] = 256 + i
            file_ids = self.merge(file_ids, max_pair, 256 + i)
        self.build_vocab()

    def encode(self, text: str):
        text_ids = list(text.encode('utf-8'))
        for key, value in self.pairs_dict.items():
            text_ids = self.merge(text_ids, key, value)
        return text_ids

    def decode(self, encoded_values: list[int]):
        value = b"".join(self.vocab[val] for val in encoded_values)
        return value.decode('utf-8')

    def merge(self, ids: list[int], pair: tuple, new_val: int):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(new_val)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def create_pairs(self, file_ids: list[int]):
        pair_freq = {}
        for pair in zip(file_ids, file_ids[1:]):
            if pair in pair_freq:
                pair_freq[pair] += 1
            else:
                pair_freq[pair] = 1
        return pair_freq

    def build_vocab(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.pairs_dict.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
