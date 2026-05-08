# Independent benchmarking via NVIDIA AIPerf

This directory adds a second, **independent** measurement of the SAYIT
generator's latency and throughput using NVIDIA AIPerf. It is the system
analogue of using two independent scorers (M² and ERRANT) to validate a
GEC model — Locust is one ground truth, AIPerf is another, and the
thesis can claim "two industrial benchmarks agree within X %".

## Why a shim is needed

AIPerf and similar industrial LLM benchmarks expect an OpenAI-compatible
`/v1/chat/completions` endpoint. The SAYIT backend exposes a domain-specific
`/api/v1/generate` endpoint with a custom JSON schema. `shim.py` is a thin
~150-line FastAPI service that translates between the two protocols.

```
AIPerf  --(/v1/chat/completions)-->  shim.py  --(/api/v1/generate)-->  SAYIT
   ^                                                                      |
   +---------- request latency, RPS, output token throughput --------------+
```

## Methodological honesty

The SAYIT backend **does not stream tokens** — it returns the full exercise
JSON in one response. This has two consequences for AIPerf:

1. **Request Latency** and **Request Throughput** from AIPerf are valid
   system metrics — they can be compared 1:1 with Locust's
   p50/p95/p99 and RPS. This is what we report in §3.4.

2. **TTFT** and **ITL** from AIPerf are NOT meaningful for this system.
   The shim's stream=True path emits one large SSE chunk after backend
   inference completes, so any TTFT measured here is just total request
   latency. Do not cite TTFT/ITL numbers from AIPerf in the thesis.

This is stated explicitly in `shim.py` for any reviewer who reads the
source.

## How to run

```bash
# 1. Install AIPerf and the shim's dependencies.
pip install aiperf fastapi uvicorn httpx

# 2. With the SAYIT generator already running on :8000, start the shim:
./run_shim.sh                # listens on :8001

# 3. From another terminal, run AIPerf at concurrency 1, 2, 4, 8, 16:
./run_aiperf.sh

# 4. Cross-validate against Locust output (assumes Locust runs are done):
python3 cross_validate.py
```

After step 4, `reports/cross_validation.md` contains a side-by-side table:

| Сценарий | Locust p50 | AIPerf p50 | Δ p50 | … |
|---|---|---|---|---|
| 1 клиент | 3 200 мс | 3 250 мс | +1.6 % | … |
| 4 клиента | 6 800 мс | 6 720 мс | −1.2 % | … |

This is the table to drop into §3.4 of the thesis.

## Files

| File | Purpose |
|---|---|
| `shim.py` | FastAPI service translating OpenAI ⇄ SAYIT |
| `run_shim.sh` | Start the shim, waits for backend readiness |
| `run_aiperf.sh` | Five-stage AIPerf sweep (concurrency 1, 2, 4, 8, 16) |
| `cross_validate.py` | Builds the side-by-side comparison table |

## What to write in the thesis

Suggested wording for §3.4 (right after the Locust paragraph):

> «Для независимой валидации замеров пропускной способности и задержек
> выполнен повторный прогон средствами NVIDIA AIPerf — индустриального
> бенчмарка инференса больших языковых моделей, реализующего методологию
> MLPerf Inference v5.1. Поскольку AIPerf требует OpenAI-совместимый
> программный интерфейс, тогда как разработанный сервис экспонирует
> предметно-ориентированный контракт, измерения проведены через
> программный адаптер, преобразующий протоколы без изменения логики
> инференса. Результаты двух независимых инструментов согласуются в
> пределах X % по медианной задержке и Y % по 99-му процентилю
> (таблица N), что подтверждает корректность представленных
> характеристик производительности.»
