import sys
import nltk
from nltk.corpus import gutenberg, stopwords
from nltk.probability import FreqDist

def main():
    # Ensure required NLTK data packages are downloaded
    nltk.download('gutenberg', quiet=True)
    nltk.download('stopwords', quiet=True)

    # Get filename from command line
    if len(sys.argv) < 2:
        print("Usage: python corpusAnalysis.py filename")
        return
    filename = sys.argv[1]

    # Load tokens from Gutenberg corpus
    tokens = gutenberg.words(filename)
    num_tokens = len(tokens)

    # Compute types (unique tokens)
    types = set(tokens)
    num_types = len(types)

    # Get stop words
    stop_words = set(stopwords.words('english'))

    # Compute types excluding stop words
    types_no_stopwords = [t for t in types if t.lower() not in stop_words]
    num_types_no_stopwords = len(types_no_stopwords)

    # Compute 10 most common tokens
    fdist = FreqDist(tokens)
    most_common = fdist.most_common(10)

    # Find long types (more than 13 characters)
    long_types = [t for t in types if len(t) > 13]

    # Find nouns ending with 'ation'
    types_ending_with_ation = [t for t in types if t.endswith('ation')]

    # POS tag the types ending with 'ation'
    nouns_ending_with_ation = []
    for t in types_ending_with_ation:
        # Tag the word in context
        tag = nltk.pos_tag([t])[0][1]
        if tag.startswith('NN'):  # Noun
            nouns_ending_with_ation.append(t)

    # Print the results
    print("Text:", filename)
    print("Tokens:", num_tokens)
    print("Types:", num_types)
    print("Types excluding stop words:", num_types_no_stopwords)
    print("10 most common tokens:", most_common)
    print("Long types:", long_types)
    print("Nouns ending in 'ation':", nouns_ending_with_ation)

if __name__ == "__main__":
    main()
