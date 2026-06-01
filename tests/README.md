# SAYIT Generator — System Test Suite

Этот набор скриптов реализует всё, что в главе 3 ВКР заявлено в разделах 3.4
и 3.5. Скрипты построены вокруг трёх индустриальных инструментов:

| Раздел | Инструмент | Зачем |
|--------|------------|-------|
| 3.4 функциональное | **pytest + Schemathesis** | smoke-тесты (ФТ-01..ФТ-14) + property-based fuzzing OpenAPI-схемы |
| 3.4 нагрузочное    | **Locust** (`LoadTestShape`) | latency / throughput / stress / endurance |
| 3.4 кросс-валидация | **NVIDIA AIPerf** + shim | независимое подтверждение latency/RPS — методология MLPerf Inference v5.1 |
| 3.5 ресурсоёмкость | **`nvidia-smi dmon`** + `analyze.py` | VRAM, мощность, токенов/с, расчёт на пользователя |

## Что нужно на хосте

- запущенный микросервис генерации (`http://localhost:8000/health` отвечает OK)
- Python 3.11 (как в самом сервисе)
- `nvidia-smi` (есть с драйвером NVIDIA)
- Docker — нужен только для замера холодного старта

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x run_all.sh load/*.sh functional/*.sh
```

## Полный прогон одной командой

```bash
./run_all.sh
```

Поэтапно (если хочется управлять):

```bash
# 1. Холодный старт (перезапускает контейнер!)
( cd load && ./measure_cold_start.sh )

# 2. Функциональные и контрактные тесты
( cd functional && ./run_functional.sh )

# 3. Распределение задержек, 10 минут (Таблица 19)
( cd load && ./run_latency.sh )

# 4. Кривая «throughput vs concurrency», 25 минут (Таблица 20, Рисунок 16)
( cd load && ./run_throughput.sh )

# 5. Стресс-тест, 25 минут (Таблица 21)
( cd load && ./run_stress.sh )

# 6. Длительный тест (8 часов по умолчанию)
( cd load && DURATION=8h ./run_endurance.sh )

# 7. Сборка таблиц для ВКР
( cd load && python analyze.py results )
```

После шага 7 в `load/results/tables/` лежат markdown-файлы с **уже
подставленными значениями** для всех таблиц 18–23 — копируй в .docx.

## Что куда попадает в текст ВКР

| Файл | Для какой таблицы / рисунка |
|------|----------------------------|
| `results/tables/table18_validity.md` | Таблица 18 — доля валидных JSON |
| `results/tables/table19_latency.md` | Таблица 19 — распределение задержек |
| `results/tables/table20_throughput.md` | Таблица 20 — пропускная способность |
| `results/tables/table21_stress.md` | Таблица 21 — поведение под стрессом |
| `results/tables/table22_resources.md` | Таблица 22 — VRAM, холодный старт |
| `results/tables/table23_per_user.md` | Таблица 23 — на одного пользователя |
| `results/latency.html` | Рисунок 15 — гистограмма задержек |
| `results/throughput.html` + Locust web UI | Рисунок 16 — кривая насыщения |
| `functional/reports/pytest-junit.xml` | Подтверждение ФТ-01..ФТ-14 |
| `functional/reports/schemathesis-junit.xml` | Подтверждение fuzz-покрытия |
| `aiperf/reports/cross_validation.md` | Сводная таблица «Locust vs AIPerf» — независимая валидация §3.4 |

## Структура каталогов

```
sayit-tests/
├── README.md                       (этот файл)
├── requirements.txt
├── run_all.sh                      end-to-end оркестратор
├── functional/
│   ├── test_smoke.py               pytest, ФТ-01..ФТ-14
│   ├── schemathesis.toml           конфиг fuzzing-а
│   └── run_functional.sh
└── load/
    ├── locustfile.py               три User-класса + два LoadShape
    ├── data/                       реалистичные payload-ы
    ├── monitor_gpu.sh              nvidia-smi dmon -> CSV
    ├── measure_cold_start.sh       замер холодного старта
    ├── run_latency.sh              шаг 3
    ├── run_throughput.sh           шаг 4
    ├── run_stress.sh               шаг 5
    ├── run_endurance.sh            шаг 6
    └── analyze.py                  сборка таблиц для ВКР
└── aiperf/
    ├── shim.py                     OpenAI-совместимый адаптер для AIPerf
    ├── run_shim.sh                 запуск shim на :8001
    ├── run_aiperf.sh               прогон AIPerf на 5 уровнях concurrency
    ├── cross_validate.py           Locust vs AIPerf side-by-side
    └── README.md                   методология независимой валидации
```

## Литературные источники для §3.4–3.5

- **Locust** — open-source distributed load-testing framework (Python),
  github.com/locustio/locust.
- **Schemathesis** — property-based testing for OpenAPI / GraphQL APIs,
  построен поверх Hypothesis. Автоматически выводит тест-кейсы из схемы
  и проверяет соответствие ответов. github.com/schemathesis/schemathesis.
- **NVIDIA System Management Interface (`nvidia-smi`)** — официальный
  инструмент мониторинга GPU; режим `dmon` собирает power, utilization,
  memory занятость с разрешением в одну секунду.
- **NVIDIA AIPerf** (бывш. GenAI-Perf) — open-source benchmark инференса
  LLM, реализующий методологию MLPerf Inference. Поддерживает любой
  OpenAI-совместимый endpoint. github.com/ai-dynamo/aiperf.
- **MLPerf Inference v5.1** — отраслевой стандарт метрик инференса
  (tokens/s, p99 TTFT/TPOT). У нас стриминга нет, поэтому из метрик
  MLPerf используем только tokens/s.
