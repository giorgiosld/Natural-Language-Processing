from nltk.corpus.reader import TaggedCorpusReader
from nltk.probability import FreqDist

def compute_task1(reader: TaggedCorpusReader):
    # Compute the number of sentences in the MIM-GOLD.sent file
    num_sentences = len(reader.sents('MIM-GOLD.sent'))

    # Get the 100th sentence in the MIM-GOLD.sent file and convert it to a string
    sentence_100 = " ".join(reader.sents('MIM-GOLD.sent')[99])
    return {
        "num_sentences": num_sentences,
        "sentence_100": sentence_100
    }

def compute_task2(reader: TaggedCorpusReader):
    # Compute the number of tokens in the MIM-GOLD.sent file
    num_tokens = len(reader.words('MIM-GOLD.sent'))

    # Compute the number of types in the MIM-GOLD.sent file
    num_types = len(set(reader.words('MIM-GOLD.sent')))

    return {
        "num_tokens": num_tokens,
        "num_types": num_types
    }

def compute_task3(reader: TaggedCorpusReader):
    # Compute the 10 most frequent tokens in the MIM-GOLD.sent file
    return FreqDist(reader.words('MIM-GOLD.sent')).most_common(10)


def compute_task4(reader: TaggedCorpusReader):
    # Compute the 20 most frequent PoS tags in the MIM-GOLD.sent file
    return FreqDist(tag for (word, tag) in reader.tagged_words('MIM-GOLD.sent')).most_common(20)


def main():
    reader = TaggedCorpusReader("./", r'.*\.sent', encoding='utf-8')
    task1 = compute_task1(reader)
    print(f"Number of sentences: {task1['num_sentences']}")
    print(f"Sentence no. 100: \n{task1['sentence_100']}\n")

    task2 = compute_task2(reader)
    print(f"Number of token: {task2['num_tokens']}")
    print(f"Number of types: {task2['num_types']}\n")

    task3 = compute_task3(reader)
    print(f"The most 10 frequent tokens \n{chr(10).join([f'{item[0]} => {item[1]}' for item in task3])}\n")

    task4 = compute_task4(reader)
    print(f"The most 20 frequent PoS tags: \n{chr(10).join([f'{item[0]} => {item[1]}' for item in task4])}\n")

if __name__ == "__main__":
    main()