import sys
import nltk
from nltk.corpus import gutenberg, stopwords
from nltk.probability import FreqDist


def download_nltk_packages():
    nltk.download('gutenberg', quiet=True)
    nltk.download('stopwords', quiet=True)


def load_tokens(filename: str):
    try:
        return gutenberg.words(filename)
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None


def compute_statistics(tokens: list):
    num_tokens = len(tokens)

    # Compute types (unique tokens)
    types = set(tokens)
    num_types = len(types)

    # Get stop words
    stop_words = set(stopwords.words('english'))

    # Compute types excluding stop words
    types_no_stopwords = {t for t in types if t not in stop_words}
    num_types_no_stopwords = len(types_no_stopwords)

    # Compute 10 most common tokens
    fdist = FreqDist(tokens)
    most_common = fdist.most_common(10)

    # Find long types (more than 13 characters)
    long_types = sorted([t for t in types if len(t) > 13], key=str.lower)

    # Find nouns ending with 'ation'
    types_ending_with_ation = [t for t in types if t.endswith('ation')]

    # POS tag the types ending with 'ation'
    nouns_ending_with_ation = []
    for t in types_ending_with_ation:
        # Tag the word in context
        tag = nltk.pos_tag([t])[0][1]
        if tag.startswith('NN'):  # Noun
            nouns_ending_with_ation.append(t)

    return {
        "num_tokens": num_tokens,
        "num_types": num_types,
        "num_types_no_stopwords": num_types_no_stopwords,
        "most_common": most_common,
        "long_types": long_types,
        "nouns_ending_with_ation": nouns_ending_with_ation
    }


def main():
    # Ensure required NLTK data packages are downloaded
    download_nltk_packages()

    # Get filename from command line
    if len(sys.argv) < 2:
        print("Usage: python corpusAnalysis.py filename")
        sys.exit(1)

    filename = sys.argv[1]

    # Load tokens from Gutenberg corpus
    tokens = load_tokens(filename)
    if tokens is None:
        sys.exit(1)

    # Compute and print statistics
    stats = compute_statistics(tokens)
    print(f"Text: {filename}")
    print(f"Tokens: {stats['num_tokens']}")
    print(f"Types: {stats['num_types']}")
    print(f"Types excluding stop words: {stats['num_types_no_stopwords']}")
    print(f"10 most common tokens: {stats['most_common']}")
    print(f"Long types: {stats['long_types']}")
    print(f"Nouns ending in 'ation': {stats['nouns_ending_with_ation']}")


if __name__ == "__main__":
    main()
