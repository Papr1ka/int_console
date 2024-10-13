#!/bin/bash

echo "создание директорий..."
mkdir -p ./bert_extra
mkdir -p ./vosk-model-small-ru-0.22/
mkdir -p ./bert_noised

echo "разархивирование..."
unzip ./models/bert_extra.zip -d ./bert_extra
unzip ./models/vosk-model-small-ru-0.22.zip -d ./vosk-model-small-ru-0.22/
unzip ./models/bert_noised.zip -d ./bert_noised

echo "Модели успешно распакованы!"
