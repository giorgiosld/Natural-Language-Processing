import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def load_pretrained_embeddings():
    """
    Load pre-trained embeddings from Gensim's online API
    """
    wiki_glove = api.load("glove-wiki-gigaword-100")
    twitter_glove = api.load("glove-twitter-100")
    return wiki_glove, twitter_glove

def find_word(a: str, b:str, x:str, glove):
    """
    Finds the most similar words to a given word using the provided word embedding model.
    """
    a = a.lower()
    b = b.lower()
    x = x.lower()
    formatted_out = f"> {a}:{b} as {x}:? \n"
    top_words = glove.most_similar_cosmul(positive=[x, b], negative=[a])
    for num, (word, score) in enumerate(top_words[:10]):
        formatted_out += f"{num + 1}: ({score:.3f}) {word} \n"
    return formatted_out

def similarity(word: str, glove):
    """
    Finds the most similar words to a given word using the provided word embedding model.
    """
    word = word.lower()
    formatted_out = f"Words most similar to '{word}': \n"
    top_words = glove.most_similar(word, topn=20)
    for num, (word, score) in enumerate(top_words):
        formatted_out += f"{num + 1}: ({score:.3f}) {word}\n"
    return formatted_out

def visualize_bias(words: list[str], model, dataset_name: str):
    """
    Visualizes word embeddings using PCA to reduce dimensionality to 2D.
    """
    vectors = []
    valid_words = []
    for word in words:
        try:
            vectors.append(model[word])
            valid_words.append(word)
        except KeyError:
            print(f"Word '{word}' not in vocabulary, skipping.")

    reduced_vectors = PCA(n_components=2).fit_transform(np.array(vectors))

    plt.figure(figsize=(10, 7))
    for i, word in enumerate(valid_words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.text(reduced_vectors[i, 0] + 0.01, reduced_vectors[i, 1] + 0.01, word, fontsize=12)
    plt.title(f"PCA Visualization of Word Embeddings ({dataset_name} GloVe)")
    plt.show()

def main():
    wiki_glove, twitter_glove = load_pretrained_embeddings()

    bias_words = ["man", "woman", "european", "african", "gay", "lesbian"]

    # Find similar words to bias words for Wikipedia GloVe
    print("Words most similar to bias words (Wikipedia GloVe):")
    [print(result) for result in [similarity(word, wiki_glove) for word in bias_words]]

    # Find similar words to bias words for Twitter GloVe
    print("\nWords most similar to bias words (Twitter GloVe):")
    [print(result) for result in [similarity(word, twitter_glove) for word in bias_words]]

    bias_relationship = [
        ("european", "white", "african"),
        ("man", "mechanic", "woman")
    ]

    # Find similar words to bias relationships
    for a, b, x in bias_relationship:
        print(f"\nIn Wikipedia glove: {find_word(a, b, x, wiki_glove)}")
        print(f"\nIn Twitter glove: {find_word(a, b, x, twitter_glove)}")

    bias_differences = ["cock", "tit" ]

    # Find similar words to bias words for Wikipedia GloVe to see differences between gloves
    print("Words most similar to bias words (Wikipedia GloVe):")
    [print(result) for result in [similarity(word, wiki_glove) for word in bias_differences]]

    # Find similar words to bias words for Twitter GloVe to see differences between gloves
    print("\nWords most similar to bias words (Twitter GloVe):")
    [print(result) for result in [similarity(word, twitter_glove) for word in bias_differences]]

    # Visualize biases using PCA
    visualize_bias(bias_words, wiki_glove, "Wikipedia")
    visualize_bias(bias_words, twitter_glove, "Twitter")

if __name__ == "__main__":
    main()