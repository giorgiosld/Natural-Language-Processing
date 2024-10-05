import gensim.downloader as api

def load_pretrained_embeddings():
    """
    Load pre-trained embeddings from Gensim's online API
    """
    wiki_glove = api.load("glove-wiki-gigaword-100")
    twitter_glove = api.load("glove-twitter-100")
    return wiki_glove, twitter_glove

def find_similar_words(a: str, b:str, x:str, glove):
    """
    Finds the most similar words to a given word using the provided word embedding model.
    """
    a = a.lower()
    b = b.lower()
    x = x.lower()
    print(f"> {a}:{b} as {x}:?")
    top_words = glove.most_similar_cosmul(positive=[x, b], negative=[a])
    for num, (word, score) in enumerate(top_words[:5]):
        print(f"{num + 1}: ({score:.3f}) {word}")
    print()

def main():
    wiki_glove, twitter_glove = load_pretrained_embeddings()

    bias_words = ["man", "woman", "european", "african", "american", "asian", "straight", "gay", "lesbian", "bisexual"]

    # Find similar words to bias words
    print(f"Words most similar to bias words (Wikipedia GloVe):\n{wiki_glove.most_similar((word for word in bias_words), topn=5)}")
    print(f"Words most similar to bias words (Twitter GloVe):\n{twitter_glove.most_similar((word for word in bias_words), topn=5)}")

    bias_relationship = [("european", "white", "african")]

    # Find similar words to bias relationships
    for a, b, x in bias_relationship:
        find_similar_words(a, b, x, wiki_glove)
        find_similar_words(a, b, x, twitter_glove)

if __name__ == "__main__":
    main()