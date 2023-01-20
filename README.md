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

More details about each annotation layer can be found in the guidelines (`AnnotationGuidelinesStoryARG.pdf`).

We create aggregated versions of the dataset, where each row represents a 'unique' experience annotated by between 1-4 annotators.
The aggregated versions are stored in `aggregated` and contains different files for tolerance levels for aggregation.
The experiences are aggregated based on their token overlap. All experiences that share a certain amount of token overlap (relative number of overlapping tokens) are merged into one group. 
Each csv file contains the following additional columns:

- `aggregatedID`: a unique identifier of the aggregated experience
- `Protagonist`: all protagonists annotated by any of the annotators
- `Functionsofpersonalexperiences`: all argumentative functions annotated by any of the annotators
- `words_max_span`: the maximum experience span (lowest start index and highest end index) based on start and end indices marked by the annotators
- `words_min_span`: the minimum overlapping span of experience words (highest start index and lowest end index)

## References

The dataset contains source documents from the following corpora:

- RegulationRoom (CDCP corpus): Joonsuk Park and Claire Cardie. 2018. **A corpus of eRulemaking user comments for measuring evaluability of arguments.** In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*, Miyazaki, Japan. European Language Resources Association (ELRA)
- Europolis: Marlène Gerber, André Bächtiger, Susumu Shikano, Simon Reber, and Samuel Rohr. 2018. **Deliberative abilities and influence in a transnational deliberative poll (europolis).** *British Journal of Political Science*, 48(4):1093–1118.
- ChangeMyView: Ryo Egawa, Gaku Morio, and Katsuhide Fujita. 2019. **Annotating and analyzing semantic role of elementary units and relations in online persuasive arguments.** In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop*, pages 422–428, Florence, Italy. Association for Computational Linguistics.

Please cite the papers if you use the dataset.