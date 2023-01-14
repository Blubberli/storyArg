import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils import class_weight
import torch
import torch.nn.functional as F
from torch import nn
from args import parse_arguments
from data import ClassificationDataset
import os

# specify GPU or CPU as device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def compute_metrics(pred: EvalPrediction):
    """This method computes the accuracy, precision, recall and f1 score for the predictions of the model. It adds a
    more detailed classification report to the metrics dictionary and returns it."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1).flatten()
    precision, recall, macro_f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, average='macro',
                                                                     zero_division=0)
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    report_dict = classification_report(y_true=labels, y_pred=preds, output_dict=True, zero_division=0)
    report_csv = classification_report(y_true=labels, y_pred=preds, zero_division=0)
    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        "report_dict": report_dict,
        "report_csv": report_csv
    }
    return results


def run_training(model, train_dataset, eval_dataset, training_args, class_weights):
    """This method trains the model and saves the predictions and classification reports for the training, validation and test set."""
    isExist = os.path.exists(training_args.output_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(training_args.output_dir)
        print("The new directory %s is created!" % training_args.output_dir)
    # set class weights for imbalanced data
    if class_weights:
        # compute class weights based on the distribution of the labels in the training data. this gives more weight to the minority class.
        class_weights = class_weight.compute_class_weight(classes=np.unique(train_dataset.labels),
                                                          y=train_dataset.labels,
                                                          class_weight="balanced")
        # a CustomTrainer object is used, that contains the class weights in the training arguments. the CustomTrainer has a different
        # compute_loss method that uses the class weights to compute the loss.
        trainer = CustomTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
    else:
        # use the standard Trainer from Huggingface
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            class_weigths=class_weights
        )

    model.to(device)
    trainer.train()
    train_results = trainer.evaluate(train_dataset)
    # evaluate on dev set
    dev_results = trainer.evaluate(eval_dataset)
    # evaluate on test set
    train_report = train_results["eval_report_csv"]
    dev_report = dev_results["eval_report_csv"]

    print("Train Report: ", train_report)
    print("Dev Report: ", dev_report)
    save_predictions(trainer, eval_dataset, training_args.output_dir, train_results, dev_results)


def save_predictions(trainer, dev_data, output_dir, train_results, dev_results):
    """This method saves the predictions of the model on the training, validation and test set.
    It also saves the classification reports for the training, validation and test set.
    It saves the original dataframe of validation and test data with the class probabilities and the predicted class."""
    dev_predictions = trainer.predict(dev_data)
    # generate probabilities over classes and save the test data with predictions as a dataframe into split directory
    dev_data.dataset['predictions'] = F.softmax(torch.tensor(dev_predictions.predictions), dim=-1).tolist()
    # generate the predicted label and add it to the dataframe
    dev_data.dataset['predicted_label'] = np.argmax(dev_predictions.predictions, axis=1).flatten()
    dev_data.dataset.to_csv(f'{str(output_dir)}/dev_df_with_predictions.csv', index=False, sep="\t")
    # save classification report for training,  validation and test set in split directory
    with open(f'{str(output_dir)}/train_report.csv', "w") as f:
        f.write(train_results["eval_report_csv"])
    with open(f'{str(output_dir)}/dev_report.csv', "w") as f:
        f.write(dev_results["eval_report_csv"])


class CustomTrainer(Trainer):
    """This class inherits from the Trainer class from Huggingface. It overrides the compute_loss method to use class weights"""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You pass the class weights when instantiating the Trainer
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).float().to(device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
    # read in arguments
    # model args: all classification details
    # data args: path to dataset etc.
    # training args: learning rate, optimizer etc.
    model_args, data_args, training_args = parse_arguments()
    print("model is trained for %s " % data_args.label)
    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # use a normal seq classification model from huggingface
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                               num_labels=model_args.labels_num)
    print("loaded model of type %s" % str(type(model)))
    train_set = pd.read_csv(data_args.traindata, sep="\t")
    val_set = pd.read_csv(data_args.testdata, sep="\t")

    train_data = ClassificationDataset(dataset=train_set,
                                       label=data_args.label, tokenizer=tokenizer, text_col=data_args.text_col)
    dev_data = ClassificationDataset(dataset=val_set,
                                     label=data_args.label, tokenizer=tokenizer, text_col=data_args.text_col)

    run_training(model=model, train_dataset=train_data, eval_dataset=dev_data,
                 training_args=training_args,
                 class_weights=training_args.class_weights)
