import glob, os
import json
from tqdm import tqdm

def wordlist_to_sentence(word_list, ignore_word = [' ','','\\'],sentence_splitter = ['\n','\r','\n\r','\r\n']):
    temp_sentence = []
    sentence = []
    for w in wordlist:
        if w in sentence_splitter:
            sentence.append(temp_sentence)
            temp_sentence = []
        elif w not in ignore_word:
            temp_sentence.append(w.lower())
    sentence.append(temp_sentence)
    temp_sentence = []
    return sentence

def text_collection_to_word_list(texts):
    wordlist = []
    for i in range(texts.count):
        tokenlist = texts.get_token_list(i)
        temp = []
        for word in tokenlist:
            temp.append(word[texts.tag_dictionary['word']].lower())
        wordlist.append(temp)
    return wordlist

def build_input(list_of_word_list, sentence_segment_func = lambda x: [x]):
    sentences = []
    for l in list_of_word_list:
        sentences.extend(sentence_segment_func(l))
    return sentences

def pantip_json_to_sentences(file_list ,ignore_word = [' ','','\\'],sentence_splitter = ['\n','\r','\n\r','\r\n']):
    temp_sentence = []
    sentence = []
    for i in tqdm(range(len(file_list[0:]))):
        file = file_list[i]
        with open(file_list[i],'r') as f:
            data = json.load(f)
        for word in data['topic']:
            for w in word:
                if w in sentence_splitter:
                    sentence.append(temp_sentence)
                    temp_sentence = []
                elif w not in ignore_word:
                    temp_sentence.append(w.lower())
            sentence.append(temp_sentence)
            temp_sentence = []
        for word in data['title']:
            for w in word:
                if w in sentence_splitter:
                    sentence.append(temp_sentence)
                    temp_sentence = []
                elif w not in ignore_word:
                    temp_sentence.append(w.lower())
            sentence.append(temp_sentence)
            temp_sentence = []
        for comment in data['comments']:
            for word in comment:
                for w in word:
                    if w in sentence_splitter:
                        sentence.append(temp_sentence)
                        temp_sentence = []
                    elif w not in ignore_word:
                        temp_sentence.append(w.lower())
                sentence.append(temp_sentence)
    return sentence

def get_file_name_list(path, regx):
    corpus_directory = glob.escape(path)
    file_list = sorted(glob.glob(os.path.join(path, regx)))
    return file_list