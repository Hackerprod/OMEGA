import numpy as np

from omega.data.loader import TimeSeriesDataLoader
from omega.data.text_loader import TextWindowDataLoader
from omega.nlp.continuous import ContinuousTextEncoder


def test_timeseries_loader_reuses_buffers():
    rng = np.random.default_rng(7)
    data = rng.standard_normal((20, 4)).astype(np.float32)
    loader = TimeSeriesDataLoader(data, window=3, batch_size=2, stride=2, dtype=np.float32)

    iterator = iter(loader)
    windows1, targets1 = next(iterator)
    windows2, targets2 = next(iterator)

    assert windows1.dtype == np.float32
    assert targets1.dtype == np.float32
    assert windows1.base is loader._window_buffer
    assert targets1.base is loader._target_buffer
    assert windows2.base is loader._window_buffer
    assert targets2.base is loader._target_buffer


def test_text_loader_memmap(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("abcdef", encoding="utf-8")

    encoder = ContinuousTextEncoder(d_model=8, smoothing=0.1)
    memmap_path = tmp_path / "encoded.dat"

    loader = TextWindowDataLoader.from_path(
        path=str(text_file),
        encoder=encoder,
        window=2,
        batch_size=1,
        stride=1,
        shuffle=False,
        dtype=np.float32,
        memmap_path=str(memmap_path),
    )

    assert loader.data.dtype == np.float32
    assert memmap_path.exists()
    batch = next(iter(loader))
    assert batch[0].shape[-1] == encoder.d
