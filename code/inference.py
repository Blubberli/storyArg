import pandas as pd
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data import ClassificationDataset
import numpy as np
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(load_path, model):
    """
    This method loads a model from a given path and initializes the trained weights from fine-tuning.
    :param load_path: the path to the trained model (ending: .pt)
    :param model: a BertForSequenceClassification model from transformers
    """
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict)


def generate_predictions(model, dataset, out_path, batch_size=57):
    """
    This method generates predictions for a given dataset.
    :param model: a SequenceClassification model from transformers
    :param dataset: a ClassificationDataset object
    :param batch_size: the batch size for the dataloader
    :return: a list of predictions
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_predictions = []
    all_probs = []
    model.eval()
    # add progress bar
    for batch in tqdm.tqdm(dataloader):
        with torch.no_grad():
            # put everything on the device
            for key, val in batch.items():
                batch[key] = val.to(device)
            outputs = model(**batch)
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            # convert logits to probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            all_probs.extend(probs)
            predictions = np.argmax(logits, axis=1).flatten()
            all_predictions.extend(predictions)
    dataset.dataset[args.label_col] = all_predictions
    dataset.dataset['%s_predictions' % args.label_col] = all_probs
    dataset.dataset.to_csv(f"{out_path}/new_df_with_predictions.csv", index=False, sep="\t")
    print(f"Predictions saved to {out_path}/new_df_with_predictions.csv")
    return dataset.dataset


def eval_predictions(dataset, prediction_col, label_col, out_path):
    """
    :param dataset: the dataset with generated predictions
    :param prediction_col: the columns that contains the predicted label
    :param label_col: the column that contains the labels
    :param out_path: the director to save the classification report
    :return:
    """
    report_csv = classification_report(y_true=dataset[label_col], y_pred=dataset[prediction_col], zero_division=0)
    with open(f'{str(out_path)}/test_report.csv', "w") as f:
        f.write(report_csv)
    print(f"Classification report saved to {out_path}/test_report.csv")
    print(report_csv)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to checkpoint')
    parser.add_argument('--pretrained_model', type=str, help='model type (e.g. bert-base-uncased)')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--text_col', type=str, help='the column name of the comment')
    parser.add_argument('--output_path', type=str, help='path to the directory to store output')
    parser.add_argument('--label_col', type=str, help='the column name of the label')
    parser.add_argument('--do_eval', action="store_true", help='whether to compute the classification report')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # use a normal seq classification model without features
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model,
                                                               num_labels=2)

    load_checkpoint(args.model_path, model)
    model.to(device)
    print("loaded trained model from %s" % args.model_path)
    test_set = pd.read_csv(args.data_path, sep="\t")
    if args.do_eval:
        test_data = ClassificationDataset(dataset=test_set,
                                          label=args.label_col, tokenizer=tokenizer, text_col=args.text_col)
        df_preds = generate_predictions(model, test_data, args.output_path)
        eval_predictions(df_preds, "predicted_label", args.label_col, args.output_path)
    else:
        # create dummy labels
        test_set["label"] = [0] * len(test_set)
        test_data = ClassificationDataset(dataset=test_set,
                                          tokenizer=tokenizer, text_col=args.text_col, label=args.label_col)
        generate_predictions(model, test_data, args.output_path)
