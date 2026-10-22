"""A tiny network that predicts, from the last 6 hourly measurements, whether the threshold is reached within 3 h.

Trained on the other stations only (leave one station out), exported to ONNX and quantized to INT8, the
format an embedded runtime would load. The station would run it once an hour.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from .fetch import ROOT

MODELS = ROOT / "models"
LAGS = 6


def net() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(LAGS, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16), torch.nn.ReLU(),
                               torch.nn.Linear(16, 1))


def train(x: np.ndarray, y: np.ndarray, epochs: int = 400) -> torch.nn.Module:
    torch.manual_seed(0)
    m = net()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    # Floods are rare: weight the positives so they count as much as the quiet hours.
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - y.sum()) / max(y.sum(), 1)))
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)[:, None]
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(m(tx), ty)
        loss.backward()
        opt.step()
    return m.eval()


class _Calib(CalibrationDataReader):
    def __init__(self, x: np.ndarray):
        self.rows = iter(x[i:i + 256] for i in range(0, min(len(x), 25_600), 256))

    def get_next(self):
        b = next(self.rows, None)
        return None if b is None else {"levels": b}


def export(m: torch.nn.Module, calib: np.ndarray, name: str) -> tuple[Path, Path]:
    MODELS.mkdir(exist_ok=True)
    fp32, int8, pre = (MODELS / f"{name}_{k}.onnx" for k in ("fp32", "int8", "pre"))
    # Sigmoid inside the graph: the station compares a probability with its operating point.
    full = torch.nn.Sequential(m, torch.nn.Sigmoid())
    torch.onnx.export(full, torch.zeros(1, LAGS), str(fp32), input_names=["levels"], output_names=["p"],
                      dynamic_axes={"levels": {0: "n"}, "p": {0: "n"}}, opset_version=17, dynamo=False)
    quant_pre_process(str(fp32), str(pre))
    # reduce_range: 7-bit weights avoid the U8S8 int16 saturation of x86 CPUs without VNNI (see onboard-inference-gate)
    quantize_static(str(pre), str(int8), _Calib(calib), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, reduce_range=True)
    pre.unlink()
    return fp32, int8


def session(path: Path) -> ort.InferenceSession:
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def predict(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    """Probability for every row; rows with a missing measurement get 0 (no decision without data)."""
    ok = ~np.isnan(x).any(1)
    p = np.zeros(len(x), np.float32)
    p[ok] = sess.run(None, {"levels": x[ok]})[0][:, 0]
    return p
