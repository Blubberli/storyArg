# StoryARG:
### a corpus of narratives and personal experiences in argumentative texts

This repository contains the data and code for the paper "StoryARG: a corpus of narratives and personal experiences in argumentative texts".

## Main Dataset

The files for the main dataset are located in the `data` folder. The dataset is stored in `storyarg.csv` which is a  tab-seperated file where each row represents 
an extracted story from one annotator with the corresponding annotations.
The file contains the following columns:

- `storyID`: a unique identifier of a story
- `file_name`: the name of the file of the source document
- `annotator`: the annotator who extracted the story
- `corpus`: the source corpus
- `document_text`: the whole text of the original document that was annotated / where the experience is extracted from
- `experience_words`: the text span of the extracted experience
- `hint_words`: can be empty. If not empty, it contains words, marked by the annotator as hints for why they extracted the span as an experience
- `tokens`: the number of tokens in the experience
- `stance`: whether the argumentative position of the document is clear or not
- `claim`: short summary of the claim / argumentative position of the document
- `Protagonist`: the protagonist of the experience
- `Protagonist2`: the second protagonist of the experience
- `Proximity`: the point of view of the experience
- `ExperienceType`: whether the story has a plot (story) or not (experiential knowledge)
- `Hypothetical`: whether the story is hypothetical or not
- `Functionsofpersonalexperiences`: the function of the experience in the document
- `Emotionalappeal`: whether the experience is appealing to emotions ('low', 'medium', 'high)
- `Effectiveness`: whether the experience makes the argument effective ('low', 'medium', 'high)

We create aggregated versions of the dataset, where each row represents a 'unique' experience annotated by between 1-4 annotators.
The aggregated versions are stored in `aggregated` and contains different files for tolerance levels for aggregation.
The experiences are aggregated based on their token overlap. All experiences that share a certain amount of token overlap (relative number of overlapping tokens) are merged into one group. 
Each csv file contains the following additional columns:

- `aggregatedID`: a unique identifier of the aggregated experience
- `Protagonist`: all protagonists annotated by any of the annotators
- `Functionsofpersonalexperiences`: all argumentative functions annotated by any of the annotators
- `words_max_span`: the maximum experience span (lowest start index and highest end index) based on start and end indices marked by the annotators
- `words_min_span`: the minimum overlapping span of experience words (highest start index and lowest end index)

## COVID-19 Discourse Analysis

The directory `covid19_discourse` contains the data and code for the COVID-19 discourse analysis.
The original dataset is the COVID-19 Vaccine News Reddit Discussions with ~34.000 user comments from the subreddit https://www.reddit.com/r/Coronavirus/
The whole dataset is publicly available https://www.kaggle.com/datasets/xhlulu/covid19-vaccine-news-reddit-discussions

The automatically annotated datasets are stored in `covid19_discourse/covid19_experiences`.
The file `human_annotated_covid19subset.tsv` contains the annotated subset of the discourse that is used to evaluate the models predictions.
The dataset contains the original columns of the Vaccine News Reddit Discussions dataset (e.g. post_id, post_authro, post_title...)
and the following additional columns:
- `experience_words`: the text of the automatically extracted experience
- `number_of_sentences`: the number of sentences in the experience
- `token_length`: the token length of the experience
- `annotated label`: human annotation for whether the extracted span is an experience or not
- `text_span`: the text span of the extracted experience
- `protagonist`: human annotation for the protagonist of the experience
- `proximity`: human annotation for the point of view of the experience
- `harm`: whether the experience contains a disclosure of harm
- `solution`: whether the experience is a search for a solution
- `clarification`: whether the experience is a clarification
- `background`: whether the experience is used to establish the speaker's background
- `personal`: whether the experience is personal (1) (based on protagonist and proximity) or general (0)

The file `covid19vaccination_stories_2sentence_20tokens.csv` contains the filtered dataset which was used for the discourse analysis. Every experience in this dataset
contains at least 2 sentences and 20 tokens. Besides the original columns, the dataset contains the following columns
- `experience_words`: the text of the automatically extracted experience
- `number_of_sentences`: the number of sentences in the experience
- `token_length`: the token length of the experience
- `named_entities`: the automatically detected named entities in the experience

The columns 'harms', 'solutions', 'clarifications', 'backgrounds', 'personal' are the same as in the human annotated dataset but based on the automatic annotation with the models.
The colums contain a 1 if they have been labeled as such by any of the three models and a 0 if not.
The columns with the suffix '_majority' contain the majority vote of the three models. If two models agree on a label, the majority vote is 1, otherwise 0.

The directory `cluster_results` contains the results of the clustering analysis which has been conducted once for all experiences classified as 'dislosure of harm'
and once for all experiences classified as 'search for a solution'. The files `cluster_results_harm.csv` and `cluster_results_solution.csv` contain the results of the clustering analysis.
The column `cluster` contains the cluster number of the experience.
The plots of the clustering analysis are stored in `clusters_harm.png` and `clusters_solution.png`.

## Models

The directory `code` contains the code for training the models on our corpus for three different purposes:

### 1. Extracting personal experiences
The task is to extract personal experiences from argumentative texts. The models are trained on the StoryARG dataset and evaluated on the human annotated subset of the COVID-19 discourse.
We train the models as a sentence classifier which predicts whether a sentence is part of a personal experience or not.
The three train / development splits to train the models are stored in `code/splits/sentence_labeling`.
The instances are the sentences from the documents in the StoryARG dataset. Each sentence is labeled as 1 if it is part of a personal experience and 0 if not.
The column storyID contains the unique identifier of the story the sentence is part of or is empty if the sentence is not part of a story.
### 2. Classifying personal vs. general experiences
The task is to classify personal experiences as personal or general.
The models are trained on the StoryARG dataset and evaluated on the human annotated subset of the COVID-19 discourse.
The column 'personal' is created based on the 'protagonist' and 'proximity' columns. If the protagonist is INDIVIDUAL and the proximity is FIRST-HAND, the experience is labeled as personal, otherwise as general.
The three train / development splits to train the models are stored in `code/splits/personal`.
The models take the 'experience_words' as input and the 'personal' column as label.
### 3. Classifying the function of personal experiences
The task is to classify the function of personal experiences. We train a model for each function. The three splits 
to train the models are stored in `code/splits/function`. The splits are created based on the aggregated StoryARG dataset.
Each experience can have multiple functions. The models take the 'experience_words' as input and the 'function' column as label.
('DISCLOSURE OF HARM', 'SEARCH FOR SOLUTION', 'CLARIFICATION', 'ESTABLISH BACKGROUND'

To train the models, run the following command:

```sh example_bash.sh```

The script calls the `train_classification.py` script which trains a transformer model as a binary classifier on any of the described tasks and stores the results in the `output_dir` directory.
The input data can be specified with the `traindata/testdata` argument and the corresponding label with the `label` argument.

To predict any of the described tasks, run the following command:

```python inference_classification.py --model_dir <model_dir> --input_file <input_file> --output_file <output_file>```

The script loads the saved model checkpoint from the `model_dir` directory and predicts the labels for the instances in the `input_file` and stores the results in the `output_file` file.
Set --do_eval to evaluate if the `input_file` contains labels.