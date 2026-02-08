# Throughput Benchmarks

## ACP Microbench (Steps per Call)

Fast microbenchmark executed with `python scripts/bench_quick.py --steps 20 --d-model 16`.

| Metric            | Baseline (pre-opt) | Actual (native kernels) | Delta |
| ----------------- | ------------------ | ----------------------- | ----- |
| `acp.step` (s)    | 2.00               | 0.46                    | 4.3× faster |
| `acp.update_operator` (s) | 0.05 | 0.002 | 25× faster |
| `lpu.forward` (s) | 0.05 | 0.0004 | 125× faster |
| `lpu.local_update` (s) | 0.08 | 0.011 | 7.3× faster |

Baseline values come from `benchmarks/baseline.json` (snapshot anterior a la optimización nativa).

## Stress Harness (steps/s)

`python scripts/run_stress.py --profiles audio_small text_small --dtype float32`

| Perfil       | Tipo  | Pasos | Tiempo (s) | Pasos/s |
| ------------ | ----- | ----- | ---------- | ------- |
| audio_small  | audio | 29,996 | 44.42      | 675 |
| text_small   | text  | 6,246  | 23.86      | 262 |

Notas:

- Los “pasos” corresponden a iteraciones individuales (`run_step`) registradas en `train_agent`.
- Para texto continuo se observan múltiples reversiones SCSI; el throughput sigue siendo estable gracias al scheduler suavizado.

## Cómo replicar

1. Generar datasets si es necesario:
   ```bash
   python scripts/generate_benchmark_data.py --profile audio_small
   python scripts/generate_benchmark_data.py --profile text_small
   ```
2. Ejecutar el estrés completo:
   ```bash
   python scripts/run_stress.py --dtype float32
   ```
3. Los resultados quedan en `benchmarks/stress_latest.json`. Para comparar con una versión previa ajuste el archivo y ejecute `bench_quick.py --baseline benchmarks/baseline.json`.
