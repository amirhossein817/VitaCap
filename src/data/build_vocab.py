import nltk
import pickle
import os
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary:
    """A simple vocabulary wrapper for tokenizing and mapping words to indices."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def __len__(self):
        return len(self.word2idx)


class VocabBuilder:
    """
    Vocabulary builder class to process COCO or Flickr annotations and generate a vocabulary.
    """

    def __init__(self, annotation_path, threshold=4, dataset_type="coco"):
        """
        Initialize the vocabulary builder.
        Args:
            annotation_path (str): Path to annotation JSON or text file.
            threshold (int): Minimum word frequency to include in the vocabulary.
            dataset_type (str): Dataset type, either 'coco' or 'flickr'.
        """
        self.annotation_path = annotation_path
        self.threshold = threshold
        self.dataset_type = dataset_type
        self.vocab = Vocabulary()

    def build(self):
        """
        Process the annotations to build the vocabulary.
        Returns:
            Vocabulary: The generated vocabulary object.
        """
        counter = Counter()

        if self.dataset_type.lower() == "coco":
            coco = COCO(self.annotation_path)
            ids = coco.anns.keys()

            for i, ann_id in enumerate(ids):
                caption = str(coco.anns[ann_id]["caption"])
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                counter.update(tokens)

                if (i + 1) % 1000 == 0:
                    print(f"[{i+1}/{len(ids)}] Tokenized the captions.")

        elif self.dataset_type.lower() == "flickr":
            assert os.path.isfile(self.annotation_path), "Annotation file not found!"
            with open(self.annotation_path, "r") as f:
                for i, line in enumerate(f):
                    if len(line.strip()) == 0:
                        continue
                    tokens = nltk.tokenize.word_tokenize(line.lower())
                    counter.update(tokens)

                    if (i + 1) % 1000 == 0:
                        print(f"[{i+1}] Processed lines.")
        else:
            raise ValueError("Unsupported dataset type. Use 'coco' or 'flickr'.")

        # Filter words by frequency threshold
        words = [word for word, cnt in counter.items() if cnt >= self.threshold]

        # Add special tokens
        self.vocab.add_word("<pad>")
        self.vocab.add_word("<start>")
        self.vocab.add_word("<end>")
        self.vocab.add_word("<unk>")

        # Add words to the vocabulary
        for word in words:
            self.vocab.add_word(word)

        print("Vocabulary built successfully.")
        print(f"Total words in vocabulary: {len(self.vocab)}")
        return self.vocab

    def save(self, output_path):
        """
        Save the vocabulary object to a .pkl file.
        Args:
            output_path (str): Path to save the .pkl file.
        """
        with open(output_path, "wb") as f:
            pickle.dump(self.vocab, f)
        print(f"Vocabulary saved to {output_path}")


# Example Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Vocabulary from COCO or Flickr Captions"
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        required=True,
        help="Path to COCO-style annotation JSON file or Flickr text file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path to save the vocabulary .pkl file.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Minimum word frequency to include in the vocabulary.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="coco",
        choices=["coco", "flickr"],
        help="Dataset type: 'coco' or 'flickr'.",
    )

    args = parser.parse_args()

    # Build and save vocabulary
    builder = VocabBuilder(args.annotation_path, args.threshold, args.dataset_type)
    vocab = builder.build()
    builder.save(args.output_path)
