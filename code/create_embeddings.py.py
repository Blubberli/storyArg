import pickle
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import argparse


def create_embeddings_from_file_sbert(file_path, text_col, model_path, output_path):
    """
    Creates embeddings from a file and saves them to a pickle file.
    :param file_path: path to the file
    :param output_path: path to the output file
    :return:
    """
    # read your corpus etc
    df = pd.read_csv(file_path, sep="\t")

    corpus_sentences = df[text_col].tolist()
    print("Encoding the corpus. This might take a while")
    model = SentenceTransformer(model_path, device='cuda')
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
    whitened = transform_and_normalize(corpus_embeddings)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    print("Storing file on disc")
    with open(output_path, "wb") as fOut:
        pickle.dump({text_col: corpus_sentences, 'embeddings': corpus_embeddings, "whitening": whitened}, fOut)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def compute_kernel_bias(vecs, k=None):
    """
    vecs = matrix (n x 768) with the sentence representations of your whole dataset
    (in the paper they use train, val and test sets)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    if k:
        return W[:, :k], -mu
    else:
        return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """
    kernel and bias are W and -mu from previous function, have them saved and pass to this function
    when inputing vecs
    vecs = vectors you want to whiten.
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


def create_embeddings_from_file_bert(file_path, text_col, model_path, output_path):
    # read your corpus etc
    df = pd.read_csv(file_path, sep="\t")
    corpus_sentences = df[text_col].tolist()

    # Load AutoModel from huggingface model repository
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to('cuda')
    print("Encoding the corpus. This might take a while")
    encoded_input = tokenizer(corpus_sentences, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    corpus_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    print("Storing file on disc")
    with open(output_path, "wb") as fOut:
        pickle.dump({text_col: corpus_sentences, 'embeddings': corpus_embeddings}, fOut)


def get_vector_representation_from_string(string_rep):
    """:return vector representation of a vector as string input,
    convert string [-5.69495466e-03  4.79496531e-02  7.24947453e] into vector"""
    x = np.array([x.strip() for x in string_rep[1:-1].split(" ")])
    # remove empty strings
    x = x[x != ""]
    # convert to floats
    return np.array([float(x) for x in x])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='path to data to generate predictions for')
    parser.add_argument('--text_col', type=str, help='name of the text column in the data')
    parser.add_argument('--output_path', type=str, help='path to save the predictions')
    parser.add_argument('--model_path', type=str, help='path to the pretrained LM')
    args = parser.parse_args()

    create_embeddings_from_file_sbert(file_path=args.file_path, text_col=args.text_col, model_path=args.model_path,
                                      output_path=args.output_path)
