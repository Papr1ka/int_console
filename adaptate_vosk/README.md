# Интеллектуальный пульт составителя

### Адаптация языковой модели

Файлы, которые имеют значение

```bash
adaptate_vosk
├── Dockerfile.kaldi-vosk-model-022-ru - Докерфайл для сборки
├── extra.txt - Файл для записи фраз
├── README.md
└── tmp - папка, в которой в будущем будут данные для адаптированной модели
```

Наполните файл **./extra.txt** фразами, к которым вы хотите, чтобы модель адаптировалась

Создание docker image

`docker build --file Dockerfile.kaldi-vosk-model-022-ru --tag alphacep/kaldi-vosk-model-022-ru:latest .`

Процесс может занять **очень значительное время** преимущественно из-за фазы компиляции.

Переменная `путь` - папка, куда будут выгружены обученые веса

`docker run -d -p 2722:2700 -v {путь}:/out ./out  alphacep/kaldi-vosk-model-022-ru:latest`

Пусть это будет папка tmp в текущей директории

`docker run -d -p 2722:2700 -v tmp:/out ./out  alphacep/kaldi-vosk-model-022-ru:latest`

В итоге мы получаем следующую файловую структуру

```bash
adaptate_vosk
├── Dockerfile.kaldi-vosk-model-022-ru
├── extra.txt
├── README.md
└── tmp
    ├── Gr.fst - полученные файлы
    └── HCLr.fst - полученные файлы
```

Файлы Gr.fst и HCLr.fst следует перенести в папку с моделью vosk, в базовом случае - vosk-model-small-ru-0.22/graph

```bash
int_console/vosk-model-small-ru-0.22/graph
├── disambig_tid.int
├── Gr.fst - новые файлы
├── HCLr.fst - новые файлы
└── phones
    └── word_boundary.int
```

Готово!
