import pymorphy3
import re
import torch
import time
import subprocess

from vosk import Model, KaldiRecognizer
from transformers import BertForSequenceClassification, BertTokenizer
import librosa
import numpy as np
import wave
import soundfile as sf
import settings

model = Model(settings.VOSK)
recognizer = KaldiRecognizer(model, 16000)

morph = pymorphy3.MorphAnalyzer(lang='ru')

clf_path = settings.BERT
classifier = BertForSequenceClassification.from_pretrained(clf_path)
tokenizer = BertTokenizer.from_pretrained(clf_path)


# Классификация команд
def classify(text):
    """
    Классификация текстовых транскрибций команд

    Args:
        text (str): Текстовая транскрибция команды

    Returns:
        int:  Предсказанный label
    """

    tokenized_text = tokenizer([text], return_tensors='pt')

    with torch.no_grad():
        outputs = classifier(**tokenized_text)

    prediction = np.argmax(outputs.logits)

    return int(prediction)


# Преобразование текстовых числительных в их численное представление
def text_to_num(words):
    """
    Преобразует текстовое представление числительного на русском языке в численное значение

    Args:
        words (list[str]): Список слов из транскрибции

    Returns:
        int: Численное значение числительного
    """

    num_dict = {
        "ноль": 0,
        "один": 1,
        "два": 2,
        "три": 3,
        "четыре": 4,
        "пять": 5,
        "шесть": 6,
        "семь": 7,
        "восемь": 8,
        "девять": 9,
        "десять": 10,
        "одиннадцать": 11,
        "двенадцать": 12,
        "тринадцать": 13,
        "четырнадцать": 14,
        "пятнадцать": 15,
        "шестнадцать": 16,
        "семнадцать": 17,
        "восемнадцать": 18,
        "девятнадцать": 19,
        "двадцать": 20,
        "тридцать": 30,
        "сорок": 40,
        "пятьдесят": 50,
        "шестьдесят": 60,
        "семьдесят": 70,
        "восемьдесят": 80,
        "девяносто": 90,
        "сто": 100
    }

    num = 0
    for word in words:
        if word in num_dict:
            num += num_dict[word]

    return num


# Извлечение числительных из текста
def get_num(line):
    """
    Извлечение числительных и преобразование в числовой формат

    Args:
        line (str): Текстовая транскрибция голосовой команды

    Returns:
        int: Численное значение атрибута или -1, если не нашлось числительных
    """

    words = line.split(' ')

    # morph = pymorphy2.MorphAnalyzer(lang='ru')
    nums = [word for word in words if 'NUMR' in morph.parse(word)[0].tag]  # Отбор числительных

    if nums:
        return text_to_num(nums)
    return -1


# Получение числовых атрибутов из команд
def get_attribute(text, label):
    """
    Извлечение атрибутов

    Args:
        text(str): Текстовая транскрибция аудио
        label (int): Предсказанный класс

    Returns:
        int: Численное значение атрибута или -1, если не нашлось числительных или у команды не предусмотрен атрибут
    """

    if label in [4, 10]:  # Только в командах 4 и 10 нужно извлекать количество вагонов
        return get_num(text)
    else:
        return -1


# Лемматизация текста
def preprocess(text):
    """
    Лемматизация текста

    Args:
        text(str): Текстовая транскрибция аудио

    Returns:
        str: Строка, состоящая из лемматизированных слов
    """
    text = text.replace('ё', 'е')
    text = re.sub(r'[^а-яА-ЯёЁ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([morph.parse(word)[0].normal_form for word in text.split(' ')])


# Функция для оценки шума (например, по первым 0.5 секундам)
def estimate_noise(y, sr, noise_duration=0.5):
    noise_samples = int(noise_duration * sr)
    noise_part = y[:noise_samples]
    stft_noise = np.abs(librosa.stft(noise_part))
    return np.mean(stft_noise, axis=1)


# Применение спектрального вычитания
def spectral_subtraction(y, sr, noise_est):
    stft_speech = librosa.stft(y)
    magnitude_speech = np.abs(stft_speech)
    phase_speech = np.angle(stft_speech)

    # Вычитание спектра шума из спектра сигнала
    magnitude_clean = np.maximum(magnitude_speech - noise_est[:, np.newaxis], 0)

    # Восстановление сигнала с очищенным спектром
    stft_clean = magnitude_clean * np.exp(1j * phase_speech)
    y_clean = librosa.istft(stft_clean)

    return y_clean


def filter_noise(input_file, output_file):
    # Загружаем аудиофайл
    data, samplerate = librosa.load(input_file, sr=None)

    # Оцениваем шум
    noise_estimation = estimate_noise(data, samplerate)

    # Применяем спектральное вычитание
    cleaned_data = spectral_subtraction(data, samplerate, noise_estimation)

    # Сохраняем очищенный аудиофайл
    sf.write(output_file, cleaned_data, samplerate)


def check_and_convert_sample_rate(input_file, target_sample_rate=16000):
    # Используем librosa для получения текущего sample rate
    data, samplerate = librosa.load(input_file, sr=None)

    if samplerate != target_sample_rate:
        output_file = "converted_audio.wav"
        
        # Команда ffmpeg для изменения sample rate
        subprocess.run([
            "ffmpeg", "-i", input_file, "-ar", str(target_sample_rate), output_file, "-y"
        ])
        
        return output_file  # Возвращаем путь к новому файлу с 16000 Hz
    else:
        return input_file  # Если всё в порядке, возвращаем исходный файл

def trans_one_audio(file_path):
    temp_cleaned_file = "cleaned_audio.wav"
    file_path = check_and_convert_sample_rate(file_path)
    filter_noise(file_path, temp_cleaned_file)
    wf = wave.open(temp_cleaned_file, "rb")

    full_text = ""

    while True:
        data = wf.readframes(4000)  # Читаем 4000 фреймов за раз
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            # Извлекаем текст и добавляем его в полный текст
            full_text += eval(result)['text'] + " "
        else:
            # Можно игнорировать частичные результаты или использовать их
            recognizer.PartialResult()

    # Финальный результат
    final_result = recognizer.FinalResult()
    full_text += eval(final_result)['text']

    # Закрываем файл
    wf.close()
    return full_text.strip()


def form_answer(audio_file):
    """
    Получение выходов модели

    Args:
        self (str): Путь к аудио файлу

    Returns:
        dict: Словарь с предсказаниями
    """

    start = time.time()
    res = {'text': trans_one_audio(audio_file)}
    print(f'Vosk: {(time.time() - start) * 1000} ms')

    start = time.time()
    res['label'] = classify(res['text'])
    print(f'Bert: {(time.time() - start) * 1000} ms')

    start = time.time()
    prep_text = preprocess(res['text'])
    res['attribute'] = get_attribute(prep_text, res['label'])
    print(f'Attributes: {(time.time() - start) * 1000} ms')

    return res
