import sys
from nltk.corpus.reader import TaggedCorpusReader

def load_file(filename: str):
    try:
        with open(filename) as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None

def main():

    filename = "MIM-GOLD.sent"
    mim_gold = load_file(filename)

    if mim_gold is None:
        sys.exit(1)

    res = TaggedCorpusReader(filename, mim_gold)

    print(f"Number of sentences: {len(res.sents(mim_gold))}")
    print(f"Sentence no. 100: {res.sents(mim_gold)[100]}")
    print(f"Number of tokens: {len(res.words(mim_gold))}")
    print(f"Number of types: {len(set(res.words(mim_gold)))}")

if __name__ == "__main__":
    main()