from __future__ import annotations
import time


import os
import argparse
import json
import psutil

RAM_STATS = []

def check_mem():
    global RAM_STATS
    pid = os.getpid()
    python_process = psutil.Process(pid)
    memoryUse = python_process.memory_info()[0]/2.**30
    RAM_STATS.append(memoryUse)


class Predictor:
    def __call__(self, audio_path: str):
        prediction = form_answer(audio_path)
        result = {
            "audio": os.path.basename(audio_path),          # Audio file base name
            "text": prediction.get("text", -1),             # Predicted text
            "label": prediction.get("label", -1),           # Text class
            "attribute": prediction.get("attribute", -1),   # Predicted attribute (if any, or -1)
        }
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get submission.")
    parser.add_argument(
        "--src",
        type=str,
        help="Путь к исходным аудио-файлам",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="Путь к выходным аудио-файлам",
    )
    args = parser.parse_args()

    if (args.src is None) or not (len(args.src) > 0 and os.path.exists(args.src) and (
        os.path.isdir(args.src))):
        print("Пожалуйста, укажите аргумент src (существующая папка)")
        exit(-1)

    if (args.dst is None) or not (len(args.dst) > 0 and os.path.exists(args.dst) and (
        os.path.isdir(args.dst))):
        print("Пожалуйста, укажите аргумент dst (существующая папка)")
        exit(-1)

    # Загружаем модели в память
    global form_answer
    from model import form_answer
    
    predictor = Predictor()
    time_stats = []
    results = []
    for audio_path in os.listdir(args.src):
        start = time.time()
        print("processing", audio_path, "...")
        result = predictor(os.path.join(args.src, audio_path))
        check_mem()
        print(f'Total time: {(time.time() - start) * 1000} ms')
        results.append(result)
        time_stats.append(time.time() - start)
    
    print("Суммарное время обработки", sum(time_stats))
    print("Максимум потребления ОЗУ", max(RAM_STATS), "Гб")

    with open(
        os.path.join(args.dst, "submission.json"), "w", encoding="utf-8"
    ) as outfile:
        json.dump(results, outfile)
    print("Готово")
