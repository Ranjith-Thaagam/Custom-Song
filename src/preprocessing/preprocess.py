from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from aksharamukha.transliterate import process
from openphonemizer import OpenPhonemizer

# Only import the classes that exist in your dp module
from dp.preprocessing.text import Preprocessor, LanguageTokenizer,SequenceTokenizer
from datasets import Dataset
import torch
import os 
from indicnlp import common, loader


# detect language
def _is_tamil(text):
    return all('\u0B80' <= c <= '\u0BFF' or c.isspace() for c in text)

def _is_hindi(text):
    return all('\u0900' <= c <= '\u097F' or c.isspace() for c in text)

def _is_malayalam(text):
    return all('\u0D00' <= c <= '\u0D7F' or c.isspace() for c in text)

def _is_kannada(text):
    return all('\u0C80' <= c <= '\u0CFF' or c.isspace() for c in text)

def _is_telugu(text):
    return all('\u0C00' <= c <= '\u0C7F' or c.isspace() for c in text)

def _is_gujarati(text):
    return all('\u0A80' <= c <= '\u0AFF' or c.isspace() for c in text)

def _is_punjabi(text):
    return all('\u0A00' <= c <= '\u0A7F' or c.isspace() for c in text)

def _is_marathi(text):
    return _is_hindi(text)  # shares Devanagari script

def _is_english(text):
    return all(('A' <= c <= 'Z') or ('a' <= c <= 'z') or c.isspace() for c in text)


def detect_language(text):
    """
    Detect the language of the given text based on Unicode ranges.
    """
    try:
        if _is_tamil(text): return 'ta'
        elif _is_hindi(text): return 'hi'
        elif _is_malayalam(text): return 'ml'
        elif _is_kannada(text): return 'kn'
        elif _is_telugu(text): return 'te'
        elif _is_gujarati(text): return 'gu'
        elif _is_punjabi(text): return 'pa'
        elif _is_marathi(text): return 'mr'
        elif _is_english(text): return 'en'
        else: return 'ta'
    except Exception as e:
        print(f"Error detecting language: {e}")
        return 'ta'
    
def normalize_text(text): 
    """
    Normalize text for Indic languages using IndicNLP normalizer.
    """
    lines = text.split('\n')
    text = ' '.join(lines)

    factory = IndicNormalizerFactory()
    normalized = ""

    for word in text.split():
        lang = detect_language(word)
        if lang == 'en':
            normalized += word + ' '
            continue
        normalizer = factory.get_normalizer(lang)
        normalized += normalizer.normalize(word) + " "

    return normalized.strip()


def preprocess_transcript(normalized):
    """
    Convert normalized text into phoneme-like transliteration.
    Tamil → Latin via Aksharamukha
    English → phonemized via OpenPhonemizer
    """

    try:
        all_transcripts = ""

        allowed_classes = [Preprocessor, LanguageTokenizer, SequenceTokenizer]
        with torch.serialization.safe_globals(allowed_classes):
            phonemizer = OpenPhonemizer(disable_gpu=False)

        for word in normalized.split():
            lang = detect_language(word)

            if lang in ['ta', 'tam']:
                transliterated = process('Tamil', 'ISO', word)
                #print(f"Tamil detected: {word} → {transliterated}")

            elif lang == 'en':
                transliterated = phonemizer.phonemizer(word, lang='en_us')
                #print(f"English detected: {word} → {transliterated}")

            elif lang == 'hi':
                transliterated = process('Devanagari', 'ISO', word)
                #print(f"Hindi detected: {word} → {transliterated}")

            elif lang == 'ml':
                transliterated = process('Malayalam', 'ISO', word)
                #print(f"Malayalam detected: {word} → {transliterated}")
            
            elif lang == 'kn':
                transliterated = process('Kannada', 'ISO', word)
                #print(f"Kannada detected: {word} → {transliterated}")
            
            elif lang == 'te':
                transliterated = process('Telugu', 'ISO', word)
                #print(f"Telugu detected: {word} → {transliterated}")

            elif lang == 'gu':
                transliterated = process('Gujarati', 'ISO', word)
                #print(f"Gujarati detected: {word} → {transliterated}")

            all_transcripts += transliterated + ' '

        return all_transcripts.strip()

    except Exception as e:
        print(f"Error preprocessing transcript: {e}")
        return None
 
def convert_hf_dataset(dataset):
    """
    Convert a dataset to a Hugging Face dataset
    """
    hf_dataset = Dataset.from_dict(dataset)
    return hf_dataset