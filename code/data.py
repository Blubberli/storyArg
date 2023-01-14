from torch.utils.data import Dataset
import torch


class ClassificationDataset(Dataset):
    """
    A Dataset for text classification
    """

    def __init__(self, dataset, label, tokenizer, text_col):
        """
        :param dataset: the dataframe, e.g. to the validation data
        :param label: the column name of the label to be predicted, e.g. "moderation_comment"
        :param tokenizer: a tokenizer object (e.g. RobertaTokenizer) that has a default for creating encodings for text
        :param text_col: the column name that stores the actual text, e.g. COMMENT
        """
        self.dataset = dataset
        self.labels = self.dataset[label].values
        self.text_col = text_col
        # drop all columns with no text
        self.dataset = self.dataset[self.dataset[self.text_col].notna()]

        self.encodings = tokenizer(list(self.dataset[self.text_col]), return_tensors='pt', padding=True,
                                   truncation=True)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).type(torch.LongTensor)
        # labels = labels.type(torch.LongTensor)
        return item

    def __len__(self):
        return len(self.labels)


class MultiLabelDataset(Dataset):
    """
    A Dataset for multi-label text classification
    """

    def __init__(self, dataset, labels, tokenizer, text_col):
        """
        :param dataset: the dataframe, e.g. to the validation data
        :param label: the column name of the label to be predicted, e.g. "moderation_comment"
        :param tokenizer: a tokenizer object (e.g. RobertaTokenizer) that has a default for creating encodings for text
        :param text_col: the column name that stores the actual text, e.g. COMMENT
        """
        self.dataset = dataset
        # create an index for labels
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for i, label in enumerate(labels)}
        # create a colum with the label vector that one-hot encodes the labels if they have 1 or 0 in the corresponding column
        self.dataset["label_vector"] = self.dataset[labels].apply(lambda x: [1 if label == 1 else 0 for label in x],
                                                                  axis=1)
        self.label_vector = self.dataset["label_vector"].values
        self.text_col = text_col
        # drop all columns with no text
        self.dataset = self.dataset[self.dataset[self.text_col].notna()]

        self.encodings = tokenizer(list(self.dataset[self.text_col]), return_tensors='pt', padding=True,
                                   truncation=True)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.label_vector[idx]).type(torch.FloatTensor)
        # labels = labels.type(torch.LongTensor)
        return item

    def __len__(self):
        return len(self.dataset)
