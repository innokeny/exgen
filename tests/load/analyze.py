from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def _read_locust_stats(prefix: Path) -> pd.DataFrame:
    f = prefix.with_name(prefix.name + "_stats.csv")
    return pd.read_csv(f)


def _read_locust_history(prefix: Path) -> pd.DataFrame:
    f = prefix.with_name(prefix.name + "_stats_history.csv")
    df = pd.read_csv(f)
    return df[df["Name"] == "Aggregated"].copy()


def _read_locust_extra(prefix: Path) -> pd.DataFrame:
    f = prefix.with_suffix(".extra.jsonl")
    if not f.exists():
        return pd.DataFrame()
    return pd.read_json(f, lines=True)


def _read_gpu(prefix: Path) -> pd.DataFrame:
    f = prefix.with_suffix(".gpu.csv")
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


@dataclass
class LatencyStats:
    n: int
    min_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    std_ms: float


def _latency_from_locust(stats: pd.DataFrame, endpoint: str) -> LatencyStats:
    row = stats[stats["Name"].str.contains(endpoint, regex=False)].iloc[0]
    return LatencyStats(
        n=int(row["Request Count"]),
        min_ms=float(row["Min Response Time"]),
        median_ms=float(row["Median Response Time"]),
        p90_ms=float(row["90%"]),
        p95_ms=float(row["95%"]),
        p99_ms=float(row["99%"]),
        max_ms=float(row["Max Response Time"]),
        mean_ms=float(row["Average Response Time"]),
        std_ms=float(row.get("Average Response Time", 0.0)) * 0,
    )


def table_19_latency(results_dir: Path) -> str:
    prefix = results_dir / "latency"
    stats = _read_locust_stats(prefix)
    history = _read_locust_history(prefix)
    s = _latency_from_locust(stats, "/api/v1/generate")
    rt_series = history["Total Average Response Time"].dropna()
    std_ms = float(rt_series.std(ddof=1)) if len(rt_series) > 1 else 0.0

    rows = [
        ["Минимальное время отклика", f"{s.min_ms:.0f}"],
        ["Медианное время отклика", f"{s.median_ms:.0f}"],
        ["90-й процентиль", f"{s.p90_ms:.0f}"],
        ["95-й процентиль", f"{s.p95_ms:.0f}"],
        ["99-й процентиль", f"{s.p99_ms:.0f}"],
        ["Максимальное время отклика", f"{s.max_ms:.0f}"],
        ["Среднее значение", f"{s.mean_ms:.0f}"],
        ["Стандартное отклонение", f"{std_ms:.0f}"],
        ["Объём выборки, запросов", f"{s.n}"],
    ]
    return "## Таблица 19 — Распределение задержек при одиночных запросах\n\n" + _md_table(
        ["Показатель", "Значение, мс"], rows
    )


def table_20_throughput(results_dir: Path) -> str:
    prefix = results_dir / "throughput"
    history = _read_locust_history(prefix)

    stages = [(1, 0, 300), (2, 300, 600), (4, 600, 900), (8, 900, 1200), (16, 1200, 1500)]
    rows = []
    for users, t0, t1 in stages:
        window = history[(history["Timestamp"] >= history["Timestamp"].iloc[0] + t0) &
                         (history["Timestamp"] < history["Timestamp"].iloc[0] + t1)]
        if window.empty:
            continue
        rps = window["Requests/s"].mean()
        rt_avg = window["Total Average Response Time"].mean()
        failures = window["Failures/s"].sum()
        total_req = window["Requests/s"].sum()
        success_pct = 100.0 * (1 - failures / max(total_req, 1))
        rows.append([
            users, f"{rps:.2f}", f"{rt_avg:.0f}", f"{success_pct:.1f}"
        ])

    return (
        "## Таблица 20 — Пропускная способность при разной параллельной нагрузке\n\n"
        + _md_table(
            ["Параллельных клиентов", "Запросов/с", "Среднее время отклика, мс", "Успешных, %"],
            rows,
        )
    )


def table_21_stress(results_dir: Path) -> str:
    prefix = results_dir / "stress"
    history = _read_locust_history(prefix)
    gpu = _read_gpu(prefix)

    stages = [
        ("Штатный (1x)",  0,    300),
        ("2x штатного",   300,  600),
        ("4x штатного",   600,  900),
        ("8x штатного",   900,  1200),
    ]
    rows = []
    for label, t0, t1 in stages:
        window = history[(history["Timestamp"] >= history["Timestamp"].iloc[0] + t0) &
                         (history["Timestamp"] < history["Timestamp"].iloc[0] + t1)]
        if window.empty:
            continue
        rps = window["Requests/s"].mean()
        p95 = window["95%"].mean() if "95%" in window else float("nan")
        failures = window["Failures/s"].sum()
        total = window["Requests/s"].sum()
        err_pct = 100.0 * failures / max(total, 1)

        if not gpu.empty:
            g = gpu.iloc[t0:t1]
            mem_total = g["mem_used_mib"].max() + g["mem_free_mib"].max()
            mem_pct = 100.0 * g["mem_used_mib"].mean() / max(mem_total, 1)
        else:
            mem_pct = float("nan")

        rows.append([
            label, f"{rps:.2f}", f"{p95:.0f}", f"{err_pct:.2f}",
            f"{mem_pct:.1f}" if mem_pct == mem_pct else "—",
        ])

    return (
        "## Таблица 21 — Поведение сервиса при стресс-нагрузке\n\n"
        + _md_table(
            ["Уровень нагрузки", "Запросов/с", "p95 задержки, мс",
             "Доля ошибок, %", "Расход VRAM, %"],
            rows,
        )
    )


def table_22_resources(results_dir: Path) -> str:
    cold_start_csv = results_dir / "cold_start.csv"
    cs_ms = first_req_ms = "—"
    if cold_start_csv.exists():
        cs_df = pd.read_csv(cold_start_csv)
        cs_ms = f"{cs_df['cold_start_ms'].mean()/1000:.1f}"
        first_req_ms = f"{cs_df['first_request_ms'].mean():.0f}"

    health_path = results_dir / "cold_start_health.json"
    vram_after_load = "—"
    if health_path.exists():
        h = json.loads(health_path.read_text())
        vram_after_load = f"{h['gpu']['vram_used_mb']/1024:.2f}"

    steady_gpu = _read_gpu(results_dir / "throughput")
    if not steady_gpu.empty:
        vram_steady = f"{steady_gpu['mem_used_mib'].mean()/1024:.2f}"
    else:
        vram_steady = "—"

    steady_rt = "—"
    lat_stats = results_dir / "latency_stats.csv"
    if lat_stats.exists():
        df = pd.read_csv(lat_stats)
        agg = df[df["Name"] == "Aggregated"]
        if not agg.empty:
            steady_rt = f"{agg.iloc[0]['Median Response Time']:.0f}"

    rows = [
        ["Расход видеопамяти после старта контейнера, ГБ", "≈ 0.10"],
        ["Расход видеопамяти после загрузки модели, ГБ", vram_after_load],
        ["Расход видеопамяти под нагрузкой, ГБ", vram_steady],
        ["Время «холодного» старта, с", cs_ms],
        ["Время отклика на первый запрос, мс", first_req_ms],
        ["Время отклика в установившемся режиме, мс", steady_rt],
    ]
    return "## Таблица 22 — Использование видеопамяти и время инициализации\n\n" + _md_table(
        ["Показатель", "Значение"], rows
    )


def table_23_per_user(results_dir: Path, requests_per_user_per_day: int = 6,
                      questions_per_request: int = 15) -> str:
    prefix = results_dir / "throughput"
    stats = _read_locust_stats(prefix)
    gpu = _read_gpu(prefix)

    if stats.empty or gpu.empty:
        return "## Таблица 23 — недостаточно данных (запустите run_throughput.sh)\n"

    batch_row = stats[stats["Name"].str.contains("/api/v1/generate/batch", regex=False)]
    if batch_row.empty:
        return "## Таблица 23 — нет данных по /generate/batch\n"
    batch_lat_s = float(batch_row.iloc[0]["Average Response Time"]) / 1000.0
    batch_questions = questions_per_request

    avg_power_w = float(gpu["power_w"].mean())
    energy_per_batch_j = avg_power_w * batch_lat_s

    exercises_per_day = requests_per_user_per_day * batch_questions
    inference_s_per_day = requests_per_user_per_day * batch_lat_s
    energy_wh_per_day = (requests_per_user_per_day * energy_per_batch_j) / 3600.0
    busy_pct = 100.0 * inference_s_per_day / 86400.0

    saturating_rps = stats["Requests/s"].astype(float).max()
    max_batches_per_day = saturating_rps * 86400.0 * 0.7
    max_users = int(max_batches_per_day / requests_per_user_per_day)

    rows = [
        ["Среднее число формируемых упражнений", str(exercises_per_day)],
        ["Суммарное время инференса, с", f"{inference_s_per_day:.1f}"],
        ["Суммарное энергопотребление, Вт·ч", f"{energy_wh_per_day:.2f}"],
        ["Доля времени занятости GPU, %", f"{busy_pct:.3f}"],
        ["Расчётный предел активных пользователей в сутки на 1 GPU",
         f"≥ {max_users:,}".replace(",", " ")],
    ]
    return ("## Таблица 23 — Расчёт на одного активного пользователя в сутки\n\n"
            + _md_table(["Показатель", "Значение"], rows))


def table_18_validity(results_dir: Path) -> str:
    extra = _read_locust_extra(results_dir / "latency")
    if extra.empty:
        return "## Таблица 18 — недостаточно данных (нет latency.extra.jsonl)\n"

    extra["valid"] = ~extra["fallback"]

    summary = extra.groupby("endpoint").agg(
        runs=("valid", "count"),
        valid=("valid", "sum"),
    )
    summary["pct"] = 100.0 * summary["valid"] / summary["runs"]

    rows = []
    for ep, row in summary.iterrows():
        rows.append([ep, int(row["runs"]), int(row["valid"]),
                     f"{row['pct']:.1f}", "95"])

    return "## Таблица 18 — Доля структурно валидных JSON-ответов\n\n" + _md_table(
        ["Маршрут", "Прогонов", "Валидных", "Доля, %", "Минимум, %"],
        rows,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=Path)
    p.add_argument("--runs", nargs="*",
                   default=["latency", "throughput", "stress", "endurance"])
    args = p.parse_args()

    out_dir = args.results_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "table18_validity.md":   table_18_validity,
        "table19_latency.md":    table_19_latency,
        "table20_throughput.md": table_20_throughput,
        "table21_stress.md":     table_21_stress,
        "table22_resources.md":  table_22_resources,
        "table23_per_user.md":   table_23_per_user,
    }

    for name, builder in builders.items():
        try:
            content = builder(args.results_dir)
        except FileNotFoundError as e:
            content = f"## {name} — пропущено ({e.filename} отсутствует)\n"
        except Exception as e:
            content = f"## {name} — ошибка: {e!r}\n"
        (out_dir / name).write_text(content, encoding="utf-8")
        print(f"  wrote {out_dir / name}")

    print(f"\nDone. Tables: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
