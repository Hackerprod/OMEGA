# Benchmark Datasets

The stress tests use synthetic but reproducible datasets to approximate audio and
continuous-text workloads. All assets are generated on the fly to avoid large
binaries in the repository.

## Audio Benchmarks

Each audio profile is defined by `(sample_rate, duration_seconds, channels)`.
Signals are generated as the sum of sinusoidal components plus mild noise to
stress the ACP update path with high-frequency dynamics.

| Name           | Sample Rate | Duration | Channels | Window | Stride | Batch |
| -------------- | ----------- | -------- | -------- | ------ | ------ | ----- |
| `audio_small`  | 16 kHz       | 30 s     | 1        | 64     | 16     | 16    |
| `audio_medium` | 32 kHz       | 180 s    | 1        | 128    | 32     | 32    |
| `audio_large`  | 44.1 kHz     | 600 s    | 2        | 256    | 64     | 64    |

Generated files are stored in `benchmarks/generated/audio_<name>.npy`.

## Continuous Text Benchmarks

Text corpora are random excerpts from Project Gutenberg-style samples with
controlled length (in characters). Encoding is performed with
`ContinuousTextEncoder` using the `d_model` listed below. Window and batch sizes
cover short, medium, and long contexts.

| Name          | Characters | d_model | Window | Stride | Batch |
| ------------- | ---------- | ------- | ------ | ------ | ----- |
| `text_small`  | 50 k       | 64      | 32     | 8      | 16    |
| `text_medium` | 250 k      | 96      | 64     | 16     | 32    |
| `text_large`  | 500 k      | 128     | 96     | 16     | 48    |

Generated files are stored in `benchmarks/generated/text_<name>.txt` and their
encoded memmaps under the same directory when running stress tests.

## Generation

Use `python scripts/generate_benchmark_data.py --profile audio_small` (or any
profile listed above) to create the corresponding dataset. The stress harness
(`scripts/run_stress.py`) will generate missing assets automatically before
running measurements.
