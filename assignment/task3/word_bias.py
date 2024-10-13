import os

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

def reduce_dimensions(vectors: list[np.ndarray], n_components: int = 2):
    """
    Reduce the dimensionality of the vectors using PCA.
    """
    return PCA(n_components=n_components).fit_transform(np.array(vectors))

def plot_embeddings(reduced_vectors: np.ndarray, words: list[str], dataset_name: str):
    """
    Plot the reduced word embeddings.
    """
    plt.figure(figsize=(10, 7))
    for i, word in enumerate(words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.text(reduced_vectors[i, 0] + 0.01, reduced_vectors[i, 1] + 0.01, word, fontsize=12)
    plt.title(f"PCA Visualization of Word Embeddings ({dataset_name} GloVe)")

def save_plot(path: str):
    """
    Save the current plot to the specified path.
    """
    save_dir = os.path.dirname(path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(path)
    print(f"Plot saved to {path}")
    plt.show()

def extract_bias_words(words: list[str], model):
    """
    Extracts word vectors and valid words from the model.
    """
    vectors = []
    valid_words = []
    for word in words:
        vectors.append(model[word])
        valid_words.append(word)

    return vectors, valid_words

def visualize_bias(words: list[str], model, dataset_name: str):
    """
    Visualizes word embeddings using PCA to reduce dimensionality to 2D.
    Optionally saves the plot to the specified path.
    """
    vectors, valid_words = extract_bias_words(words, model)

    reduced_vectors = reduce_dimensions(vectors)
    plot_embeddings(reduced_vectors, valid_words, dataset_name)
    save_path = f"resources/{dataset_name.lower()}_bias.png"
    save_plot(save_path)

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

    bias_differences = ["cock", "tits" ]

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