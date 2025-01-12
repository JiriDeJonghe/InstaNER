from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NERDataset(Dataset):
    """
    Represents a dataset for NER for the transformer type model.
    This dataset extendes the pytorch Dataset class, providing a __getitem__ method to tokenize the sentences for training.
    """

    def __init__(
        self,
        data: list[str],
        tokenizer: AutoTokenizer,
        label2id: dict[str, str],
        max_length=128,
    ):
        """
        Initializes the dataset.

        Args:
            data (list[str]): list of the original sentences
            tokenizer (AutoTokenizer): tokenizer for the model
            label2id (list[str, int]): label of mappings from label to id

        Fields:
            data (list[str]): list of the original sentences
            tokenizer (AutoTokenizer): tokenizer for the model
            label2id (list[str, int]): label of mappings from label to id
            max_length (int): maximum length of the sentences
            converted_sentences (list[dict[str, list[str | int]]]): list of dataset samples converted to a format that can be used in training
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.converted_sentences = self.convert_sentences(
            self.data, self.label2id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Tokenizes the sentences to a subword level tokenization.
        Creates new labels for this new subword-tokenization according to the following rules:
        1. All tokens will be mapped to their corresponding word using word_ids()
        2. The label -100 will be assigned to special tokens [CLS] and [SEP] and None
        3. Only the first token of a word should be labeled, other subtokens get assigned -100

        Args:
            idx (int): id of the example in the dataset

        Returns:
            dict[str, list[str|int]: the encoded sentences with the added labels and word_ids
        """
        tokens = self.converted_sentences[idx]["tokens"]
        ner_tags = self.converted_sentences[idx]["ner_tags"]

        encoding = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True)
        word_ids = encoding.word_ids()

        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(ner_tags[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        encoding["labels"] = label_ids

        return encoding

    def convert_sentences(self, sentences: list[str], label2id: list[str, str]):
        """
        Converts the format of the sentences in a json where eacah element has the following structure:
        {
            "id": str,
            "ner_tags": list[int],
            "tokens": list[str]
        }

        Args:
            sentences (list[str]): the sentences to format to JSON format
            label2id (list[str, str]): label of mappings from label to id

        Returns:
            list[dict[str, list[str | int]]]: list of dataset samples in JSON format
        """
        converted_sentences = []
        for id, sentence in enumerate(sentences):
            try:
                sentence_dict = {}
                sentence_dict["id"] = id
                ner_tags, tokens = self.convert_sentence(sentence, label2id)
                sentence_dict["ner_tags"] = ner_tags
                sentence_dict["tokens"] = tokens
                converted_sentences.append(sentence_dict)
            except Exception:
                continue

        return converted_sentences

    def convert_sentence(self, sentence: str, label2id: list[str, str]):
        """
        Converts the format of the sentence in a format that is able to be used in training by generating the ner_tags and tokens list

        Args:
            sentence (str): the sentence for which to generate the ner_tags and tokens for
            label2id (list[str, int]): label of mappings from label to id

        Returns:
            list[int]: list of NER tags for this sentence
            list[str]: list of the words of this sentence
        """
        ner_tags = []
        tokens = []

        for token, label in sentence:
            tokens.append(token)
            ner_tag = label2id[label.upper()]
            ner_tags.append(int(ner_tag))

        return ner_tags, tokens

    def get_converted_sentences(self) -> dict[str, str | list[int]]:
        """
        Retrieves the converted sentences by providing the dictionary containing "tokens" and "ner_tags" fields.
        Tokens are the subwords of the sentence and labels are the corresponding NER tags.

        Returns:
            dict[str, str | list[int]]: dictionary containing the "tokens" and "ner_tags" fields
        """
        tokens = []
        labels = []

        for sentence in self.converted_sentences:
            tokens.append(" ".join(sentence["tokens"]))
            labels.append(sentence["ner_tags"])

        return {"tokens": tokens, "ner_tags": labels}
