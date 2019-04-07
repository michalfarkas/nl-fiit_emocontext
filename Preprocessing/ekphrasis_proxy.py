from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

class EkphrasisProxy():
    def __init__(self, **kwargs):
        self.text_processor = TextPreProcessor(
            omit=kwargs.get('normalize',[]),
            normalize=kwargs.get('normalize', ['url', 'email', 'phone', 'user', 'time', 'url', 'date']),
            annotate=kwargs.get('annotate', {}),
            fix_html=kwargs.get('fix_html', True),
            segmenter=kwargs.get('segmenter',"twitter"),
            corrector=kwargs.get('corrector',"twitter"),
            unpack_hashtags=kwargs.get('unpack_hashtags',True),
            unpack_contractions=kwargs.get('unpack_contractions',True),
            spell_correct_elong=kwargs.get('fix_elongation',True),
            spell_correction=kwargs.get('spell_correction',True),
            fix_bad_unicode=kwargs.get('fix_bad_unicode',True),
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        )

    def preprocess_text(self, text):
        return self.text_processor.pre_process_doc(text)


