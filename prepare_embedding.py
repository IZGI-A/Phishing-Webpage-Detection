import trafilatura as tr
from sentence_transformers import SentenceTransformer
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import torch
import numpy as np
import os
import pickle
import sys
from googletrans import Translator


def extract_text():
    """
    Extract the text from the html files in the benign and phishing folders
    :return: None
    """
    # Get the html files from the benign and phishing folders
    benign_mislead_path = sys.argv[1]
    benign_mislead_files = os.listdir(benign_mislead_path)
    phishing_path = sys.argv[2]
    phishing_files = os.listdir(phishing_path)
    os.makedirs('extracted_benign_misleading_text', exist_ok=True)
    os.makedirs('extracted_phishing_text', exist_ok=True)

    print("*****benign_mislead_files*****", len(benign_mislead_files))

    # Get the text from the benign and misleading html files
    for file in benign_mislead_files:
        try:
            with open(benign_mislead_path + "/" + file, 'r', encoding="utf-8") as f:
                html = f.read()
                text = tr.extract(html)
                if text is None:
                    continue
                with open('extracted_benign_misleading_text/' + file[:-4] + '.txt', 'w+', encoding="utf-8") as f:
                    print("utf-8 ---", file[:-4])
                    f.write(text)
        except UnicodeDecodeError:
            with open(benign_mislead_path + "/" + file, 'r', encoding="windows-1256") as f:
                html = f.read()
                text = tr.extract(html)
                if text is None:
                    continue
                with open('extracted_benign_misleading_text/' + file[:-4] + '.txt', 'w+', encoding="utf-8") as f:
                    print("windows-1256 ---", file[:-4])
                    f.write(text)

    print("*****phishing_files*****", len(phishing_files))
    # Get the text from the phishing html files
    for file in phishing_files:
        try:
            with open(phishing_path + "/" + file, 'r', encoding="utf-8") as f:
                html = f.read()
                text = tr.extract(html)
                if text is None:
                    continue
                with open('extracted_phishing_text/' + file[:-4] + '.txt', 'w+', encoding="utf-8") as f:
                    print("utf-8 ---", file[:-4])
                    f.write(text)
        except UnicodeDecodeError:
            with open(phishing_path + "/" + file, 'r', encoding="windows-1256") as f:
                html = f.read()
                text = tr.extract(html)
                if text is None:
                    continue
                with open('extracted_phishing_text/' + file[:-4] + '.txt', 'w+', encoding="windows-1256") as f:
                    print("windows-1256 ---", file[:-4])
                    f.write(text)


def translate_text(files, path, chunk_size=4000, dest_language='en'):
    translator = Translator()
    print('-' * 50, "translate_text", '-' * 50)
    for file in files:
        try:
            with open(path + "/" + file, 'r', encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path + "/" + file, 'r', encoding="windows-1256") as f:
                text = f.read()
        print(path + "/" + file, end=" ")
        chunks = [text[i:(i + chunk_size if i + chunk_size < len(text) else len(text) - 1)] for i in
                  range(0, len(text), chunk_size)]
        translated_chunks = []
        for chunk in chunks:
            # print("chunk", chunk)
            try:
                if chunk == "" or chunk == " " or chunk == "\n" or chunk == "\t" or chunk == "\r":
                    continue
                translated = translator.translate(chunk.strip(), dest=dest_language)
                translated_chunks.append(translated.text)
            except:
                continue
        translated_text = ' '.join(translated_chunks)
        with open('translated_text/' + path + "/" + file[:-4] + '.txt', 'w+', encoding="utf-8") as f:
            print("---> translated_text" + path + "/" + file[:-4] + '.txt')
            f.write(translated_text)


def create_translated_text():
    benign_mislead_path = "extracted_benign_misleading_text/"  # extracted_benign_misleading_text
    benign_mislead_files = os.listdir(benign_mislead_path)
    phishing_path = "extracted_phishing_text/"  # extracted_phishing_text
    phishing_files = os.listdir(phishing_path)
    os.makedirs('translated_text/' + benign_mislead_path, exist_ok=True)
    os.makedirs('translated_text/' + phishing_path, exist_ok=True)

    translate_text(benign_mislead_files, benign_mislead_path)
    translate_text(phishing_files, phishing_path)


def generate_sbert_embeddings_arrays(files, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists('translated_text/' + path):
        create_translated_text()
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to(device)
    sentence_bert_embeddings = np.array([]).reshape(0, 768)
    sentence_bert_embeddings_labels = np.array([]).reshape(0, 1)
    print("-" * 50, "Sentence BERT", "-" * 50)
    if path == "extracted_phishing_text/":
        translator = Translator()
    for file in files:
        with open('translated_text/' + path + "/" + file, 'r', encoding="utf-8") as f:
            text = f.read()
        if len(text) == 0 or text == " " or text == "\n" or text == "\t" or text == "\r":
            continue
        print(path + "/" + file)
        sentence_embeddings = model.to(device).encode(text)
        sentence_embeddings = sentence_embeddings.reshape(1, -1)
        sentence_bert_embeddings = np.concatenate((sentence_bert_embeddings, sentence_embeddings))

        if path == "extracted_benign_misleading_text":
            sentence_bert_embeddings_labels = np.concatenate(
                (sentence_bert_embeddings_labels, np.zeros((len(sentence_embeddings), 1))))

        elif path == "extracted_phishing_text/":
            sentence_bert_embeddings_labels = np.concatenate(
                (sentence_bert_embeddings_labels, np.ones((len(sentence_embeddings), 1))))

    result = np.concatenate((sentence_bert_embeddings, sentence_bert_embeddings_labels), axis=1)
    return result, sentence_bert_embeddings_labels


def generate_xlm_roberta_embeddings_arrays_without_mean_pooling(files, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('aditeyabaral/sentencetransformer-xlm-roberta-base').to(device)
    xlm_roberta_embeddings = np.array([]).reshape(0, 768)
    xlm_roberta_embeddings_labels = np.array([]).reshape(0, 1)
    print("-" * 50, "XLM-Roberta-Without-Mean-Pooling", "-" * 50)
    for file in files:
        try:
            with open(path + "/" + file, 'r', encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path + "/" + file, 'r', encoding="windows-1256") as f:
                text = f.read()
        print(path + "/" + file)
        sentence_embeddings = model.to(device).encode(text)
        sentence_embeddings = sentence_embeddings.reshape(1, -1)
        xlm_roberta_embeddings = np.concatenate((xlm_roberta_embeddings, sentence_embeddings))

        if path == "extracted_benign_misleading_text":
            xlm_roberta_embeddings_labels = np.concatenate(
                (xlm_roberta_embeddings_labels, np.zeros((len(sentence_embeddings), 1))))

        elif path == "extracted_phishing_text/":
            xlm_roberta_embeddings_labels = np.concatenate(
                (xlm_roberta_embeddings_labels, np.ones((len(sentence_embeddings), 1))))

    result = np.concatenate((xlm_roberta_embeddings, xlm_roberta_embeddings_labels), axis=1)
    return result, xlm_roberta_embeddings_labels


def generate_xlm_roberta_embeddings_arrays(files, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0))
    model_name = 'xlm-roberta-base'
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaModel.from_pretrained(model_name).to(device)
    model.eval().to(device)

    xlm_roberta_embeddings = np.array([]).reshape(0, 768)
    xlm_roberta_embeddings_labels = np.array([]).reshape(0, 1)

    print("-" * 50, "XLM-Roberta", "-" * 50)
    for file in files:
        print(path + "/" + file)
        try:
            with open(path + "/" + file, 'r', encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path + "/" + file, 'r', encoding="windows-1256") as f:
                text = f.read()
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input.to(device))
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).to(device)
            xlm_roberta_embeddings = np.concatenate((xlm_roberta_embeddings, sentence_embeddings.to('cpu').numpy()))
            if path == "extracted_benign_misleading_text":
                xlm_roberta_embeddings_labels = np.concatenate(
                    (xlm_roberta_embeddings_labels, np.zeros((len(sentence_embeddings), 1))))

            elif path == "extracted_phishing_text/":
                xlm_roberta_embeddings_labels = np.concatenate(
                    (xlm_roberta_embeddings_labels, np.ones((len(sentence_embeddings), 1))))

    result = np.concatenate((xlm_roberta_embeddings, xlm_roberta_embeddings_labels), axis=1)
    return result, xlm_roberta_embeddings_labels


def create_transformers_embeddings(transformers_name):
    if not os.path.exists('extracted_benign_misleading_text') or not os.path.exists('extracted_phishing_text'):
        extract_text()
    benign_mislead_path = "extracted_benign_misleading_text"
    benign_mislead_files = os.listdir(benign_mislead_path)
    phishing_path = "extracted_phishing_text/"
    phishing_files = os.listdir(phishing_path)
    os.makedirs('embeddings', exist_ok=True)

    if transformers_name == "xlm-roberta":
        result_legitimate, label_leg = generate_xlm_roberta_embeddings_arrays(benign_mislead_files, benign_mislead_path)
        # print("result_legitimate shape", result_legitimate.shape)

        result_phishing, label_phis = generate_xlm_roberta_embeddings_arrays(phishing_files, phishing_path)
        # print("result_phishing shape", result_phishing.shape)

        result = np.concatenate((result_legitimate, result_phishing), axis=0)
        # print("result shape", result.shape)
        # print("result labels", result[:-1, :])
        with open('embeddings/' + 'embeddings-xlm-roberta.pkl', 'wb') as file:
            pickle.dump(result, file)

    elif transformers_name == "xlm-roberta-without-mean-pooling":
        result_legitimate, label_leg = generate_xlm_roberta_embeddings_arrays_without_mean_pooling(benign_mislead_files,
                                                                                                   benign_mislead_path)
        result_phishing, label_phis = generate_xlm_roberta_embeddings_arrays_without_mean_pooling(phishing_files,
                                                                                                  phishing_path)
        result = np.concatenate((result_legitimate, result_phishing), axis=0)
        with open('embeddings/' + 'embeddings-xlm-roberta.pkl', 'wb') as file:
            pickle.dump(result, file)

    elif transformers_name == "sbert":
        result_legitimate, label_leg = generate_sbert_embeddings_arrays(benign_mislead_files, benign_mislead_path)

        result_phishing, label_phis = generate_sbert_embeddings_arrays(phishing_files, phishing_path)

        result = np.concatenate((result_legitimate, result_phishing), axis=0)
        # print("result shape", result.shape)
        # print("result labels", result[:-1, :])
        with open('embeddings/' + 'embeddings-sbert.pkl', 'wb') as file:
            pickle.dump(result, file)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(
            "Usage: python prepare_embedding.py <path to the benign and misleading html files> <path to the phishing html files>")
        exit(1)
    # Parameters:
    # 1. Path to the benign and misleading html files
    # 2. Path to the phishing html files
    # extract_text()
    create_transformers_embeddings("xlm-roberta-without-mean-pooling")
    # create_transformers_embeddings("xlm-roberta")
    # create_transformers_embeddings("sbert")
