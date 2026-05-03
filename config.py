import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SLANG_DICT_PATH = os.path.join(DATA_DIR, "slang_dict.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "seq2seq_model.pt")
VOCAB_CACHE_PATH = os.path.join(BASE_DIR, "models", "vocab_cache.pkl")
