import re

import Preprocessing.regex_expressions as regex
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer

tokenizer = WordTokenizer()

class Preprocessing(object):
    """
    Module for text pre-processing
    """

    def __init__(self, **kwargs):
        self.normalize = kwargs.get('normalize', True)
        self.expand = kwargs.get('expand', True)
        self.escape_punctuation = kwargs.get('escape_punctuation', True)
        self.negation = kwargs.get('negation', True)

    def process_test(self, text):
        if self.normalize:
            text = self.normalization(text)
        if self.expand:
            text = self.phrase_expanding(text)
        if self.escape_punctuation:
            text = self.escaping(text)
        if self.negation:
            text = self.word_negation(text)

        # return text.lower().split()

        tokens = tokenizer.tokenize(text)
        return [t.text for t in tokens]

    def normalization(self, text):
        text = re.sub(regex.INVISIBLE_REGEX, '', text)
        text = re.sub(regex.EMAIL_REGEX, ' :email: ', text)
        text = re.sub(regex.URL_REGEX, ' :url: ', text)
        text = re.sub(regex.USER_REGEX, ' @USER ', text)

        text = re.sub(regex.QUOTATION_REGEX, '\"', text)
        text = re.sub(regex.APOSTROPHE_REGEX, '\'', text)
        text = re.sub(r"\s+", " ", text)

        return text

    def phrase_expanding(self, text):
        text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
        text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
        text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
        text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
        text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)

        return text

    def word_negation(self, text):
        text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]as|[Ww]ould)n't", r"\1\2 not", text)
        text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
        text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
        text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)

        return text

    def escaping(self, text):

        # text = re.sub(r"[‼.,;:?!…]+", r" \1 ", text)
        text = re.sub(r"[-]+", r" - ", text)
        text = re.sub(r"[_]+", r" _ ", text)
        text = re.sub(r"[=]+", r" = ", text)
        text = re.sub(r"[\&]+", r" \& ", text)
        text = re.sub(r"[\+]+", r" \+ ", text)

        return text
