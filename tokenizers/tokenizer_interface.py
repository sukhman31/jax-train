from abc import ABC, abstractmethod

class Tokenizer(ABC):
    @abstractmethod
    def ingest_docs(self, file_path: str):
        pass

    @abstractmethod
    def train(self, text: str):
        pass

    @abstractmethod
    def encode(self, encoded_values: str):
        pass

    @abstractmethod
    def decode(self, encoded_values: list[int]):
        pass

    def read_file(self, file_path: str):
        with open(file_path, 'r', encoding='UTF-8') as content:
            return content.read()
