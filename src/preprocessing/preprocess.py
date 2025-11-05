from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator


from openphonemizer import OpenPhonemizer
# Only import the classes that exist in your dp module
from dp.preprocessing.text import Preprocessor, LanguageTokenizer,SequenceTokenizer
from datasets import Dataset
import torch


# detect language

def _is_tamil(text):
    return all('\u0B80' <= c <= '\u0BFF' or c.isspace() for c in text)

def _is_hindi(text):
    pass

def _is_malayalam(text):
    pass

def _is_kannada(text):
    pass

def _is_telugu(text):
    pass

def _is_gujarati(text):
    pass

def _is_punjabi(text):
    pass

def _is_marathi(text):
    pass

def _is_english(text):
    return all(('A' <= c <= 'Z') or ('a' <= c <= 'z') or c.isspace() for c in text)

        

def detect_language(text):
    """
    Detect the language of the given text
    """
    try:
        if _is_tamil(text):
            return 'ta'
        elif _is_hindi(text):
            return 'hi'
        elif _is_malayalam(text):
            return 'ml'
        elif _is_kannada(text):
            return 'kn'
        elif _is_telugu(text):
            return 'te'
        elif _is_gujarati(text):
            return 'gu'
        elif _is_punjabi(text):
            return 'pa'
        elif _is_marathi(text):
            return 'mr'
        elif _is_english(text):
            return 'en'
        else:
            return 'ta'

    except Exception as e:
        print(f"Error detecting language: {e}")
        return 'ta'
    
def normalize_text(text,lang=None):
    """
    Normalize Tamil text using Indic NLP normalizer
    """

    text_line  = text.split('\n')
    text = ' '.join(text_line)

    normalized_text =""
    factory = IndicNormalizerFactory()

    for word in text.split():
        lang = detect_language(word)
        if lang == 'en':
            normalized_text += word + ' '
            continue
        normalizer = factory.get_normalizer(lang)
        normalized_text += normalizer.normalize(word)+" "
        
    return normalized_text.strip()

def preprocess_transcript(normalized):

    try:
        """
        Convert normalized Tamil text into phonemes using OpenPhonemizer
        Handles PyTorch 2.6+ safe loading
        """
        phonemized_text = None

        # Allowlist only the existing classes
        allowed_classes = [Preprocessor, LanguageTokenizer, SequenceTokenizer]

        with torch.serialization.safe_globals(allowed_classes):
            phonemizer = OpenPhonemizer(disable_gpu=False)

        phonemized_text = phonemizer.phonemizer(normalized, lang='en_us')
        
        return phonemized_text
    
    except Exception as e:
        print(f"Error preprocessing transcript: {e}")
        return None


def convert_hf_dataset(dataset):
    """
    Convert a dataset to a Hugging Face dataset
    """
    hf_dataset = Dataset.from_dict(dataset)
    return hf_dataset
