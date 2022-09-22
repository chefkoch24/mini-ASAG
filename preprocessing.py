import os
import utils
from transformers import AutoTokenizer
import xml.etree.ElementTree as et

# Settings
########################################################################################################################
# language
LANG = 'both'  # 'en' or 'ger' or 'both'
MAX_TOKENS = 256*2
OUTPUT_LENGTH = 126
########################################################################################################################


def preprocessing(path, file, tokenizer, language="en", with_questions=True, scores=True):
    """
    Preprocessor for SAF dataset.
    :param path: String / path to the folder containing the part of the dataset to preprocess
    :param file: String / save location for the preprocessed data
    :param tokenizer: HuggingFace tokenizer to preprocess the text sequences from the data set
    :param language: String / language of the dataset
    :param with_questions: Boolean / true: the questions are included as part of the text sequences
    :param scores: Boolean / true: use annotated scores as labels; false: use verification feedback as labels
    :return: None / The files are saved as .npy file at the given file location
    """
    data = []

    for files in os.listdir(path):
        if files.endswith('.xml'):
            root = et.parse(path + '/' + files).getroot()
            question = root.find('questionText').text.replace("\n", " ")
            # get reference and student answers from the files
            ref_answers = [x for x in root.find('referenceAnswers')]
            stud_answers = [x for x in root.find('studentAnswers')]

            if len(ref_answers) == 1:
                ref = ref_answers[0].text.strip()
                for x in stud_answers:
                    # arrange the text sequences according to the set parameters
                    response = x.find('response').text.strip()
                    feedback = x.find('response_feedback').text.strip()

                    label = x.find('verification_feedback').text
                    if label == "Correct":
                        label = 0
                    elif label == "Partially correct":
                        label = 1
                    elif label == "Incorrect":
                        label = 2

                    # lowercase data
                    data.append([
                        tokenizer(response.lower(), ref.lower(), truncation=True, max_length=MAX_TOKENS).input_ids,
                        tokenizer(response.lower(), ref.lower(), truncation=True, max_length=MAX_TOKENS).attention_mask,
                        tokenizer(response.lower(), ref.lower(), truncation=True, max_length=MAX_TOKENS).token_type_ids,
                        label
                    ])
    utils.save(file, data)


def preprocess_whole_set(language):
    """
    preprocess dataset to receive all variants: w or w/o questions, with scores or verification feedback
    :param language: String / language of dataset
    :return: None
    """
    if language == 'ger':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # paths to the xml formatted datasets
        train = 'data/training/german'
        ua = 'data/unseen_answers/german'
        uq = 'data/unseen_questions/german'
        folder = 'preprocessed/german'
    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # paths to the xml formatted datasets
        train = 'data/training/english'
        ua = 'data/unseen_answers/english'
        uq = 'data/unseen_questions/english'
        folder = 'preprocessed/english'
    # with scores and questions
    preprocessing(train, folder + '/wq_score_train', tokenizer, language=language)
    preprocessing(ua, folder + '/wq_score_ua', tokenizer, language=language)
    preprocessing(uq, folder + '/wq_score_uq', tokenizer, language=language)
    # with scores and w/o questions
    preprocessing(train, folder + '/score_train', tokenizer, language=language, with_questions=False)
    preprocessing(ua, folder + '/score_ua', tokenizer, language=language, with_questions=False)
    preprocessing(uq, folder + '/score_uq', tokenizer, language=language, with_questions=False)
    # with verification feedback and questions
    preprocessing(train, folder + '/wq_ver_train', tokenizer, language=language, scores=False)
    preprocessing(ua, folder + '/wq_ver_ua', tokenizer, language=language, scores=False)
    preprocessing(uq, folder + '/wq_ver_uq', tokenizer, language=language, scores=False)
    # with verification feedback and w/o questions
    preprocessing(train, folder + '/ver_train', tokenizer, language=language, scores=False, with_questions=False)
    preprocessing(ua, folder + '/ver_ua', tokenizer, language=language, scores=False, with_questions=False)
    preprocessing(uq, folder + '/ver_uq', tokenizer, language=language, scores=False, with_questions=False)


if __name__ == "__main__":
    if LANG == 'both':
        preprocess_whole_set('ger')
        preprocess_whole_set('en')
    else:
        preprocess_whole_set(LANG)
