#!/usr/bin/env python3

import pandas as pd
import scipy
import pathlib
import pickle
import argparse
from sentence_transformers import SentenceTransformer

# Encode the data with BERT Embeddings


def _sentence_encode(sentences, path_to_pretrained_model):

    model = SentenceTransformer(path_to_pretrained_model)
    pkl_file = pathlib.Path("sentence_encode.pkl")
    if pkl_file.exists():
        print("Encoded file already exists ... Loading file ...")
        with open("sentence_encode.pkl", "rb") as input_file:
            sentence_embeddings = pickle.load(input_file)
    else:
        print("Training....")
        sentence_embeddings = model.encode(sentences, show_progress_bar=True)
        with open("sentence_encode.pkl", "wb") as output_file:
            pickle.dump(sentence_embeddings, output_file)

    return sentence_embeddings, model


# Encode the Query with BERT Embeddings


def _query_encode(query, model):
    queries = [query]
    query_embeddings = model.encode(queries)

    return queries, query_embeddings


# Semantic Search Engine with BERT embeddings and input Query


def _search_embeddings(
    sentences, queries, matches, model, sentence_embeddings, query_embeddings
):

    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist(
            [query_embedding], sentence_embeddings, "cosine"
        )[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        final_search_result = []
        final_search_cosine = []

        for idx, distance in results[0:matches]:
            final_search_result.append(sentences[idx].strip())
            final_search_cosine.append(1 - distance)

    return final_search_result, final_search_cosine


# Special Print function with Cosine value for each input


def _print_with_cosine(search_results, cosine_value):
    for x, y in zip(search_results, cosine_value):
        print(x, y)
        print()


def main():

    parser = argparse.ArgumentParser(
        description="Search Engine takes Query and Model Path as Input"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path for saved pre-trained BERT model",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Type the query you want to search based on semantic meaning",
    )

    args = parser.parse_args()

    print("Search Engine with BERT Embeddings \n")
    df = pd.read_csv("trump_insult_tweets_2014_to_2021.csv")
    # model_path = r"C:\Users\karth\Downloads\bert-base-nli-mean-tokens"
    sentences = df.tweet.to_list()
    print("Encoding the Data with BERT Embeddings \n")

    sentence_embeddings_1, model_1 = _sentence_encode(sentences, args.path)

    print("Encoding the Query \n")
    query_1, query_embeddings_1 = _query_encode(args.query, model_1)
    search_results_1, cosine_value_1 = _search_embeddings(
        df.tweet.to_list(),
        query_1,
        5,
        model_1,
        sentence_embeddings_1,
        query_embeddings_1,
    )
    print("Printing the Top Inputs related to the Query \n")
    _print_with_cosine(search_results_1, cosine_value_1)


if __name__ == "__main__":
    main()
