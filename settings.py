import os

VOSK = "vosk-model-small-ru-0.22"
assert os.path.exists(VOSK), f"Ошибка, путь к модели VOSK не найден"

BERT = "bert_extra"
assert os.path.exists(BERT), f"Ошибка, путь к модели BERT не найден"

USE_NOISE_REDUCE=True
