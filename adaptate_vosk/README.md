# Интеллектуальный пульт составителя

### Адаптация языковой модели

Наполните файл **./extra.txt** фразами, к которым вы хотите, чтобы модель адаптировалась

Создание docker image

`docker build --file Dockerfile.kaldi-vosk-model-022-ru --tag alphacep/kaldi-vosk-model-022-ru:latest .`

Процесс может занять **очень значительное время** преимущественно из-за фазы компиляции.

Переменная `путь` - папка, куда будут выгружены обученые веса

`docker run -d -p 2722:2700 -v {путь}:/out ./out  alphacep/kaldi-vosk-model-022-ru:latest`

Пусть это будет папка tmp в текущей директории

`mkdir tmp`

`docker run -d -p 2722:2700 -v tmp:/out ./out  alphacep/kaldi-vosk-model-022-ru:latest`

В итоге мы получаем следующую файловую структуру

```bash

```

Файлы ... следует перенести в папку с моделью

```bash

```

Готово!
