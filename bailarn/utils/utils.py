"""
Utilities Function
"""

import glob
import math
import os
import re
import string
import gensim
import sys
import io
import time
from functools import reduce
from tqdm import tqdm, tqdm_notebook

import numpy as np
from keras.utils.np_utils import to_categorical

# from utils import co


def get_embedding_matrix(word2vec_model, word_index, fasttext=False):

    total_file_count = len(word_index)

    # Assign zeros to padding index (0)
    embedding_matrix = np.zeros(
        (len(word_index), 300))  # co.EMBEDDING_SIZE

    word_index_list = [(k, v) for k, v in word_index.items()]

    for idx in tqdm(range(len(word_index))):

        word, i = word_index_list[idx]
        if fasttext:
            embedding_matrix[i] = np.asarray(
                word2vec_model[word], dtype='float32')
        else:
            embedding_matrix[i] = np.asarray(
                word2vec_model.predict(word), dtype='float32')

    time.sleep(1)
    print("Finish")

    return embedding_matrix


def build_word_index(text_collection, word2vec_vocab=None):

    # check type of corpus --> Text collection

    sentences = set()

    if not word2vec_vocab is None:
        print("Load words from word2vec_model")
        time.sleep(1)
        vocab_list = [(k, v) for k, v in word2vec_vocab.items()]
        for idx in tqdm(range(len(vocab_list))):
            vocab = vocab_list[idx][0]
            sentences.add(vocab)

    print("Load words from text collection")
    time.sleep(1)
    for corpus_idx in tqdm(range(text_collection.count)):
        # concern that tokenize function is None in text collection
        if text_collection.tokenize_function is not None:
            for token in text_collection.get_token_list(corpus_idx):
                sentences.add(token)
        else:
            for token in text_collection.get_token_list(corpus_idx):
                sentences.add(token[0])

    word_index = build_tag_index(list(sentences), start_index=1)
    word_index["<PAD>"] = 0
    if word2vec_vocab is None:
        # if get vocab from word2vec model, UNK word is already provided
        word_index["UNK"] = len(word_index)  # co.UNK_WORD = "UNK"

    time.sleep(1)
    print("Finish")

    return word_index


class Text(object):
    def __init__(self, path, filename, content, token_list):
        self.path = path
        self.filename = filename
        self.content = content
        self.token_list = token_list

    def __str__(self):
        return """
        path: {}
        filename: {}
        content: {}
        """.format(self.path, self.filename, self.content)


class TextCollection(object):
    """
    Corpus Manager
    This class provide files loading mechanism and keep files in memory to optimize read performance
    Args:
        corpus_directory (str): relative path to corpus directory
        text_mode (bool): is files contain ground truth
        word_delimeter (str): the character that separate each words
        tag_delimeter (str): the character that separate word and tags
    limitation :
        this object is purpose for BEST 2010 corpus. Changing tag delimeter or word delimeter may cause an error.
    """

    def __init__(self, corpus_directory=None, tokenize_function=None, word_delimiter='|', tag_delimiter='/', tag_dictionary={'word': 0, 'pos': 1, 'ner': 2}):
        # Global variable

        self.corpus_directory = corpus_directory
        self.word_delimiter = word_delimiter
        self.tag_delimiter = tag_delimiter
        self.tokenize_function = tokenize_function
        self.tag_dictionary = tag_dictionary
        self.__corpus = list()

        if not corpus_directory is None:
            # Load corpus to memory
            self._load()

    def _load(self):
        """Load text to memory"""

        if not os.path.isdir(self.corpus_directory):
            raise ValueError('The corpus directory ' + self.corpus_directory +
                             ' does not exist')

        corpus_directory = glob.escape(self.corpus_directory)
        file_list = sorted(glob.glob(os.path.join(corpus_directory, "*.txt")))

        for idx in tqdm(range(len(file_list))):
            time.sleep(0.1)
            path = file_list[idx]
            with open(path, "r", encoding="utf8") as text:
                # Read content from text file
                content = text.read()

                # Preprocessing
                content = self._preprocessing(content)

                # Tokenize content
                token_list = self._tokenize(
                    content) if not self.tokenize_function is None else None

                # Create text instance
                text = Text(path, os.path.basename(path), content, token_list)

                # Add text to corpus
                self.__corpus.append(text)

        time.sleep(1)
        print("Finish")

    def _preprocessing(self, content):
        """Text preprocessing"""

        # Replace name entity '<NE>...</NE>', '<POEM>...</POEM>' and '<AB>...</AB>' symbol
        # In case of BEST free version dataset
        repls = {'<NE>': '', '</NE>': '', '<AB>': '',
                 '</AB>': '', '<POEM>': '', '</POEM>': ''}
        content = reduce(lambda a, kv: a.replace(*kv), repls.items(), content)

        # Remove new line
        content = re.sub(r"(\r\n|\r|\n)+", r"", content)

        # Convert one or multiple non-breaking space to space
        content = re.sub(r"(\xa0)+", r"\s", content)

        # Convert multiple spaces to only one space
        content = re.sub(r"\s{2,}", r"\s", content)

        # Trim whitespace from starting and ending of text
        content = content.strip(string.whitespace)

        if self.word_delimiter and self.tag_delimiter:
            # Trim word delimiter from starting and ending of text
            content = content.strip(self.word_delimiter)

            # Convert special characters (word and tag delimiter)
            # in text's content to escape character
            find = "{0}{0}{1}".format(re.escape(self.word_delimiter),
                                      re.escape(self.tag_delimiter))
            replace = "{0}{2}{1}".format(re.escape(self.word_delimiter),
                                         re.escape(self.tag_delimiter),
                                         re.escape("\t"))  # co.ESCAPE_WORD_DELIMITER
            content = re.sub(find, replace, content)

            find = "{0}{0}".format(re.escape(self.tag_delimiter))
            replace = "{1}{0}".format(re.escape(self.tag_delimiter),
                                      re.escape("\t"))  # co.ESCAPE_WORD_DELIMITER
            content = re.sub(find, replace, content)

        # Replace distinct quotation mark into standard quotation
        content = re.sub(r"\u2018|\u2019", r"\'", content)
        content = re.sub(r"\u201c|\u201d", r"\"", content)

        return content

    def _tokenize(self, content):
        return self.tokenize_function(content)

    def add_text(self, content, path="", filename="", token_list=None, mode=0):

        if mode == 0:
            # Preprocessing
            content = self._preprocessing(content)

            # Tokenize content
            token_list = self._tokenize(
                content) if not self.tokenize_function is None else None

        # Create text instance
        text = Text(path, filename, content, token_list)

        # Add text to corpus
        self.__corpus.append(text)

    def get_label_from_file(self, file_path):
        # Read ground_truth answers from a .lab file corresponding to the corpus idx
        # file_path = ./1.txt
        filename = file_path[:-4] + '.lab'

        if not os.path.exists(filename):
            raise ValueError("Answer file " + filename + " does not exist")

        with io.open(filename, 'r') as f:
            answers = {line.rstrip('\n') for line in f}
        return answers

    @property
    def count(self):
        return len(self.__corpus)

    def get_filename(self, index):
        return self.__corpus[index].filename

    def get_path(self, index):
        return self.__corpus[index].path

    def get_content(self, index):
        return self.__corpus[index].content

    def get_text(self, index):
        return self.__corpus[index]

    def get_token_list(self, index):

        if self.__corpus[index].token_list is not None:
            return self.__corpus[index].token_list
        elif self.tag_delimiter is not None and self.word_delimiter is not None:
            self.__corpus[index].token_list = []
            content = self.get_content(index)

            token_list = content.split(self.word_delimiter)

            for idx, token in enumerate(token_list):
                # Empty or Spacebar
                if token == "" or token == " ":  # co.SPACEBAR
                    word = " "  # co.SPACEBAR
                    pos_tag = "PU"
                    ner_tag = "O"
                    datum = [word, pos_tag, ner_tag]
                # Word
                else:
                    # Split word and tag by tag delimiter
                    datum = token.split(self.tag_delimiter)

                # Replace token with word and tag pair
                self.__corpus[index].token_list.append(datum)

            return self.__corpus[index].token_list
        elif self.tag_delimiter is None and self.word_delimiter is not None:
            """ BEST free version and THwiki """
            self.__corpus[index].token_list = []
            content = self.get_content(index)

            self.__corpus[index].token_list = content.split(
                self.word_delimiter)

            return self.__corpus[index].token_list
        else:
            raise Exception(
                'tag or word delimeter is missing and No tokenize is given')


class InputCollection(object):

    def __init__(self, x, readable_x, y=None, readable_y=None):
        self.x = x
        self.y = y
        self.readable_x = readable_x
        self.readable_y = readable_y


def build_input(text_collection, word_index, tag_index, sequence_length, target, for_train=True):
    """
        target:
            - pos
            - ner
            - tokenizer
            - sentiment
            - categorization
    """

    # check target is valid

    print('Start building input...')
    kwargs = dict(
        text_collection=text_collection,
        word_index=word_index,
        tag_index=tag_index,
        target=target,
    )

    if for_train:
        x, y, readable_x, readable_y = generate_x_y(**kwargs)

        if target.lower()in ['sentiment', 'categorization']:
            x = pad(x, sequence_length, 0, mode=1)
            readable_x = pad(readable_x, sequence_length, '<NULL>', mode=1)
        else:
            x = pad(x, sequence_length, 0, mode=0)
            y = pad(y, sequence_length, 0, mode=0)
            readable_x = pad(readable_x, sequence_length, '<NULL>', mode=0)
            readable_y = pad(readable_y, sequence_length, '<NULL>', mode=0)

        # Change to np.array
        x = np.asarray(x)
        if target.lower() == 'tokenizer':
            y = to_categorical(y, num_classes=len(tag_index) + 2)
        elif target.lower() == 'ner':
            y = to_categorical(y, num_classes=len(tag_index))
        elif target.lower() == 'pos':
            temp = []
            for j in y:
                t = []
                for i in j:
                    t.append([i])
                temp.append(t)
            y = np.array(temp)
        else:
            y = np.asarray(y)

        return InputCollection(x=x, y=y, readable_x=readable_x, readable_y=readable_y)
    else:
        x, readable_x = generate_x(text_collection, word_index)

        if target.lower()in ['sentiment', 'categorization']:
            x = pad(x, sequence_length, 0, mode=1)
            readable_x = pad(readable_x, sequence_length, '<NULL>', mode=1)
        else:
            x = pad(x, sequence_length, 0, mode=0)
            readable_x = pad(readable_x, sequence_length, '<NULL>', mode=0)

        # Change to np.array
        x = np.asarray(x)
        return InputCollection(x=x, readable_x=readable_x)


def pad(x, sequence_length, pad_with, mode=0):
    """
      Thinking of new name of this param.
      mode = 0 : ตัดคำที่เกินไปเป็น data ใหม่
      mode = 1 : ตัดคำที่เกินออก
    """
    paded_x = []
    c = 0
    if mode == 0:
        for i in range(len(x)):
            temp = x[i]
            c += len(x[i])
            if len(x[i]) >= sequence_length:
                steps = int(len(temp) / sequence_length)
                if(len(temp) % sequence_length != 0):
                    temp.extend(
                        [pad_with] * ((sequence_length * (steps + 1)) - len(temp)))
                else:
                    steps -= 1
                for step in range(steps + 1):
                    paded_x.append(
                        temp[step * sequence_length:(step + 1) * sequence_length])
            else:
                x[i].extend([pad_with] * (sequence_length - len(x[i])))
                paded_x.append(x[i])
    else:
        for i in range(len(x)):
            temp = x[i]
            c += len(x[i])
            if len(x[i]) >= sequence_length:
                paded_x.append(x[i][:sequence_length])
            else:
                x[i].extend([pad_with] * (sequence_length - len(x[i])))
                paded_x.append(x[i])
    # print('raw count', c)
    return paded_x


def generate_x_y(**kwargs):

    text_collection = kwargs['text_collection']
    word_index = kwargs['word_index']
    tag_index = kwargs['tag_index']
    target = kwargs['target']

    if not isinstance(word_index, dict):
        raise RuntimeError('Word index is required to be dictionary.')

    if (tag_index is not None) & (not isinstance(tag_index, dict)):
        raise RuntimeError('Tag index is required to be dictionary.')

    x = []
    readable_x = []
    y = []
    readable_y = []

    print("Start generating x y...")
    total_file_count = text_collection.count
    time.sleep(0.5)
    for corpus_idx in tqdm(range(text_collection.count)):
        fx = []
        freadable_x = []
        fy = []
        freadable_y = []
        token_list = text_collection.get_token_list(corpus_idx)

        if target.lower() in ['sentiment', 'categorization']:
            # compute x
            for token in token_list:
                try:
                    word_index[token]
                except KeyError:
                    fx.append(word_index['UNK'])  # co.UNK_WORD
                else:
                    fx.append(word_index[token])
                freadable_x.append(token)

            # compute y
            labels = text_collection.get_label_from_file(
                text_collection.get_path(corpus_idx),
            )

            # check if labels exist in tag_index
            try:
                for label in labels:
                    tag_index[label]
            except KeyError:
                raise KeyError("Tag {} from path: {} not in tag_index.".format(
                    label, text_collection.get_path(corpus_idx)))

            for tag in [tag for tag, idx in sorted(tag_index.items(), key=lambda x:x[1], reverse=False)]:
                if tag in labels:
                    fy.append(True)
                    freadable_y.append(tag)
                else:
                    fy.append(False)

        elif target.lower() == 'tokenizer':
            for token in token_list:
                index_x = text_collection.tag_dictionary['word']
                index_y = text_collection.tag_dictionary['pos']

                word = token[index_x]
                for idx, char in enumerate(word):
                    # compute x
                    try:
                        word_index[char]
                    except KeyError:
                        fx.append(1)  # co.UNK_CHAR_INDEX
                    else:
                        fx.append(word_index[char])
                    freadable_x.append(char)

                    # compute y
                    if idx == len(word) - 1:
                        try:
                            tag_index[token[index_y]]
                        except KeyError:
                            raise KeyError("Index key error!")
                        else:
                            tag = tag_index[token[index_y]]
                    else:
                        tag = 1
                    fy.append(tag)

        else:  # pos, ner
            for token in token_list:
                index_x = text_collection.tag_dictionary['word']
                word = token[index_x]
                if word == ' ' or word == '':
                    pass
                    # fy.append(tag_index["PU"])
                    # freadable_y.append('<space>')
                    # fx.append(word_index[word])
                    # freadable_x.append(word)
                else:
                    index = text_collection.tag_dictionary[target]
                    try:
                        tag_index[token[index]]
                        word_index[word]
                        freadable_x.append(word)
                        freadable_y.append(token[index])
                    except KeyError:
                        pass
                        # incorrect tag, make logs
                        # raise KeyError("Index key error!")
                    else:
                        fy.append(tag_index[token[index]])
                        fx.append(word_index[word])
        x.append(fx)
        y.append(fy)
        readable_x.append(freadable_x)
        readable_y.append(freadable_y)
    return x, y, readable_x, readable_y


def generate_x(text_collection, word_index):

    if not isinstance(word_index, dict):
        raise RuntimeError('Word/Char index is required to be dictionary.')

    print("Start generating x...")
    x = []
    readable_x = []
    total_file_count = text_collection.count
    time.sleep(0.5)
    for corpus_idx in tqdm(range(text_collection.count)):

        fx = []
        freadable_x = []
        token_list = text_collection.get_token_list(corpus_idx)
        for word in token_list:
            try:
                word_index[word]
            except KeyError:
                fx.append(word_index['UNK'])  # co.UNK_WORD
            else:
                fx.append(word_index[word])
            freadable_x.append(word)
        x.append(fx)
        readable_x.append(freadable_x)
    return x, readable_x


def build_tag_index(lst, start_index=1, reverse=False):

    index = dict()
    # Create index dict (reserve zero index for non element in index)
    for idx, key in enumerate(lst, start_index):
        # Duplicate index (multiple key same index)
        if isinstance(key, list):
            for k in key:
                if reverse:
                    index[idx] = k
                else:
                    index[k] = idx

        # Unique index
        else:
            if reverse:
                index[idx] = key
            else:
                index[key] = idx

    return index


def save_labels(prediction, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for i, pred in enumerate(prediction, 1):
        filename = str(i).zfill(6) + ".lab"
        file = open(os.path.join(path, filename), 'w')
        file.write(pred)
        file.close()


# Word_Embedder utils
def clean_sentence(cleaned_sentence):
    cleaned_sentence = re.sub(r" ", "", cleaned_sentence)
    cleaned_sentence = re.sub(r"\n", "", cleaned_sentence)
    cleaned_sentence = re.sub(r"[\d\u0E50-\u0E59]", "0", cleaned_sentence)
    repls = {'<NE>': '', '</NE>': '', '<AB>': '',
             '</AB>': '', '<POEM>': '', '</POEM>': ''}
    cleaned_sentence = reduce(lambda a, kv: a.replace(
        *kv), repls.items(), cleaned_sentence)
    return cleaned_sentence


def find_low_freq_words(all_sentences, threshold=5):
    count_dict = {}
    for sentence in all_sentences:
        for word in sentence:
            try:
                count_dict[word]
            except KeyError:
                count_dict[word] = 1
            else:
                count_dict[word] += 1

    low_freq_words = {}
    for key, value in count_dict.items():
        if value < threshold:
            low_freq_words[key] = 'UNK'
    return low_freq_words


def save_word_tag(x, prediction, target, path):
    if not os.path.exists(path):
        os.makedirs(path)
    i = 1
    for f1, f2 in zip(x, prediction):
        txt = ""
        for w, t in zip(f1, f2):
            if target == 'ner':
                txt += (str(w) + '//' + str(t) + '|')
            elif target == 'pos':
                txt += (str(w) + '/' + str(t) + '/|')
            else:
                raise ValueError('target is not recognized')

        filename = str(i).zfill(6) + ".txt"
        file = open(os.path.join(path, filename), 'w')
        file.write(txt)
        file.close()
        i += 1
