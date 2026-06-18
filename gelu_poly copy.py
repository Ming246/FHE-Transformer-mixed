"""
GeLU 多项式近似：系数 + 实现（来源 nolinear/gelu/gelu_minmax_coeffs_C{80,160}.json）。

单段/复合均先近似 y(t)，GELU(x) = x * (0.5 + y(t))；复合 y≈f1+f2(f1)，单段 y≈f(t)。

用法：
    from gelu_poly import GeluPolyEvaluator, gelu_config_for_layer
    y = GeluPolyEvaluator(layer_idx=0, level=1)(x)
"""
from __future__ import annotations

import torch

# 层 9，10 的 GeLU 输入范围较大，使用 C=160；其余层使用 C=80。
GELU_LARGE_INPUT_LAYERS = (9, 10)

GELU_SCHEME_BY_C = {
    80: {"high": "composite_27_27", "mid": "composite_15_31", "low": "single_d47"},
    160: {"high": "composite_31_31", "mid": "composite_15_31", "low": "single_d47"},
}

GELU_COEFF_BANK: dict[int, dict[str, dict]] = {
    80: {  # C=80
        "high": {  # composite_27_27
            "kind": "composite",
            "C": 80.0,
            "d1": 27,
            "d2": 31,
            "f1_coeffs":
            [
                0.0, 8.867501386153993, 0.0, -389.2941585395423,
                0.0, 9156.17177850814, 0.0, -119073.86664352781,
                0.0, 946665.0774681353, 0.0, -4919911.826562304,
                0.0, 17432377.79089243, 0.0, -43140765.17136398,
                0.0, 75291506.1118784, 0.0, -92250117.70632526,
                0.0, 77631466.6208724, 0.0, -42720298.29749595,
                0.0, 13840986.01713353, 0.0, -2001610.0266551275
            ],
            "f2_coeffs":
            [
                0.0, 2.598695436560066, 0.0, -47.04381318249947,
                0.0, 530.9329293997798, 0.0, -4564.725874870162,
                0.0, 29823.83196776347, 0.0, -135719.4919740853,
                -7.073633321929606e-06, 243731.9569913782, 9.325815586392857e-05, 2238524.030028225,
                -0.0008335565764534612, -25645379.42185455, 0.005198517527478574, 145109272.0411549,
                -0.02289074797560116, -545969559.8848352, 0.07091244975691562, 1438548742.26497,
                -0.1512416350613436, -2638640987.652164, 0.2114117187570863, 3222003691.159256,
                -0.174319995554139, -2359158992.883974, 0.06424491399423489, 784458655.5667074
            ],
            "gelu_max_err": 3.704679e-05,
        },
        "mid": {  # composite_15_31
            "kind": "composite",
            "C": 80.0,
            "d1": 15,
            "d2": 31,
            "f1_coeffs":
            [
                0.0, 5.198875992814287, 0.0, -76.12322453356832,
                0.0, 572.0504372915302, 0.0, -2226.6160280668146,
                0.0, 4789.268758549, 0.0, -5746.778634897859,
                0.0, 3601.2789651780467, 0.0, -917.8528179262662
            ],
            "f2_coeffs":
            [
                0.0, 5.135114048314053, 0.0, -238.34068096306086,
                0.0, 8118.039050119543, 0.0, -209361.36951657658,
                0.0, 4085762.2391276793, -1.1119779044503875e-06, -60436506.8042527,
                1.736374252594132e-05, 678164039.156992, -0.0001845982764892092, -5776013077.966264,
                0.0013737667502483225, 37309801710.29599, -0.007256261114943902, -182038276909.95245,
                0.027236204696231267, 664579152868.9216, -0.07173556629932666, -1783302173725.505,
                0.12852630987990185, 3408280614515.115, -0.14732578516443662, -4386335633014.4824,
                0.09540110875245869, 3404388258034.456, -0.025480945946605126, -1203172620030.193
            ],
            "gelu_max_err": 0.002577390537339852,
        },
        "low": {  # single_d47
            "kind": "single",
            "C": 80.0,
            "degree": 47,
            "f_coeffs":
            [
                0.44992136245553027, 39.9999999999988, 634.1806732001277, 0.0,
                -40887.831120141964, 0.0, 1843121.783713221, 1.0570673929462894e-06,
                -52031678.04664814, -2.724665443868947e-05, 977127498.8449755, 0.00047840933658568094,
                -12887755244.046867, -0.006041614597775465, 124394543025.96216, 0.056826866757784616,
                -906032025126.9791, -0.40831097054996496, 5095156353317.448, 2.2848757165657263,
                -22503647719187.57, -10.106877790534853, 79038661280093.95, 35.73279573110344,
                -222676103079794.2, -101.75883169541846, 505893546161953.2, 234.5010394566085,
                -928823718045537.2, -438.01082733046655, 1376706558769360.2, 662.1108887167154,
                -1639783584207246.0, -805.93302281958, 1555632562621171.8, 782.6328330757102,
                -1158447729860865.2, -597.3490864360307, 661905102447196.8, 350.1654207278581,
                -279913233862998.0, -152.0306407299061, 82490967565822.5, 46.01972643859078,
                -15117695930744.018, -8.664899499801793, 1296755846152.3208, 0.7636799622456704
            ],
            "gelu_max_err": 0.44991668356986625,
        },
    },
    160: {  # C=160
        "high": {  # composite_31_31
            "kind": "composite",
            "C": 160.0,
            "d1": 31,
            "d2": 31,
            "f1_coeffs":
            [
                0.0, 10.257851224310876, 0.0, -593.5750528689418,
                0.0, 18357.60050485964, 0.0, -315346.1658272946,
                0.0, 3340572.7468634523, 0.0, -23431139.409559347,
                0.0, 114037929.36676686, 0.0, -396992293.70797926,
                0.0, 1006765788.4104319, 0.0, -1875303627.5333767,
                -1.0498117272448213e-06, 2562139543.747525, 1.1133018302807549e-06, -2535741950.025462,
                0.0, 1768121373.6770325, 0.0, -823313263.7941971,
                0.0, 229709494.39847124, 0.0, -29034855.518375654
            ],
            "f2_coeffs":
            [
                0.0, 5.215889026706187, 0.0, -247.7431149110748,
                0.0, 8639.764485413521, 0.0, -227357.2419204267,
                0.0, 4509841.581851093, 0.0, -67573404.9266803,
                1.0681028232855493e-05, 766063797.8784915, -0.00013173087667928365, -6579738242.917668,
                0.0011606483960915654, 42805594677.26982, -0.007396223995205727, -210163540384.70984,
                0.034116040315686276, 771603234563.6464, -0.11266079947951822, -2081313370063.3018,
                0.25935026809888045, 3997397112521.6367, -0.39487434962636164, -5168597248510.093,
                0.35708042429881914, 4029608053922.8057, -0.1451093713653649, -1430362080199.5054
            ],
            "gelu_max_err": 0.005752416686675588,
        },
        "mid": {  # composite_15_31
            "kind": "composite",
            "C": 160.0,
            "d1": 15,
            "d2": 31,
            "f1_coeffs":
            [
                0.0, 5.226333363995224, 0.0, -76.84991944169633,
                0.0, 578.5620181797304, 0.0, -2254.2369748093965,
                0.0, 4851.796231694201, 0.0, -5824.397415564434,
                0.0, 3651.1101493702813, 0.0, -930.7852736348135
            ],
            "f2_coeffs":
            [
                0.0, 10.316697540640813, 0.0, -1213.5201244351988,
                0.0, 87102.23020394925, 0.0, -3802026.3590994403,
                0.0, 106940209.92008142, 0.0, -2040261790.1979957,
                4.647711646663456e-06, 27404261815.039314, -5.452161069139403e-05, -265758495795.90485,
                0.00047861916261069786, 1889412375779.5781, -0.003146280578338764, -9910956931208.477,
                0.015283501294894712, 38258079373935.04, -0.0536419007198919, -107245761907437.27,
                0.13146258643132117, 212213978095788.16, -0.21253990893547262, -280846500623393.22,
                0.20315254229138957, 222971647540966.56, -0.08680182837459585, -80275657479736.27
            ],
            "gelu_max_err": 0.5055032182639279,
        },
        "low": {  # single_d47
            "kind": "single",
            "C": 160.0,
            "degree": 47,
            "f_coeffs":
            [
                1.026795097940834, 79.9999999999979, 1224.9601331963252, 0.0,
                -77109.67621217707, 0.0, 3444950.172491895, 2.0403852117067997e-06,
                -96808581.16742048, -5.4087901410644626e-05, 1813042691.540152, 0.0009590038665453413,
                -23869657158.193695, -0.012092449259704777, 230097430049.54968, 0.11291053088630895,
                -1674319615911.781, -0.8035238993604888, 9408738739668.29, 4.452229145787691,
                -41530941705672.35, -19.514976674335678, 145797818181421.1, 68.45142266939197,
                -410594533557334.75, -193.66075171016828, 932510983469255.2, 443.96262650312065,
                -1711609201948692.0, -825.9429236752812, 2536331998964870.0, 1244.8863029621962,
                -3020357363540181.5, -1512.29835983408, 2864818501609230.0, 1466.831380083112,
                -2133016083486610.5, -1118.9781696512118, 1218565356456314.8, 655.9576379845627,
                -515250661326905.25, -284.9295157707105, 151827068319717.75, 86.3200863653668,
                -27821527207081.316, -16.271193518796405, 2386219271924.15, 1.4360040282893147
            ],
            "gelu_max_err": 1.0267729060786484,
        },
    },
}

GELU_LEVEL_KEYS = ("low", "mid", "high")

GELU_C_DEFAULT = 80.0
GELU_C_LARGE_LAYER = 160.0

GELU_POLY_FUNC_NAMES: list[str] = [
    "poly_gelu_low",
    "poly_gelu_mid",
    "poly_gelu_high",
]

# TODO(临时): 高阶 composite 的大系数在 float32 Horner 下会因大数相消算错 f1。
# 多项式求值暂用 float64，结果再 cast 回输入 dtype。
# 后续需针对系数过大做专门处理（如 Horner 缩放、重拟合、低阶方案等）。
_POLY_EVAL_DTYPE = torch.float64


def gelu_scale_for_layer(layer_idx: int) -> float:
    return GELU_C_LARGE_LAYER if layer_idx in GELU_LARGE_INPUT_LAYERS else GELU_C_DEFAULT


def gelu_config_for_layer(layer_idx: int, level: int) -> dict:
    c_scale = int(gelu_scale_for_layer(layer_idx))
    return GELU_COEFF_BANK[c_scale][GELU_LEVEL_KEYS[level]]


@torch.jit.script
def horner_poly_jit(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """TorchScript Horner；coeffs 与 x 同 device/dtype。"""
    out = torch.zeros_like(x)
    n = coeffs.numel()
    for i in range(n - 1, -1, -1):
        out = out * x + coeffs[i]
    return out


def _coeffs_to_f64(coeffs: list, device: torch.device) -> torch.Tensor:
    return torch.tensor(coeffs, device=device, dtype=_POLY_EVAL_DTYPE)


def _composite_gelu_from_t(
    x: torch.Tensor, cfg: dict, f1_f64: torch.Tensor, f2_f64: torch.Tensor
) -> torch.Tensor:
    """composite: GELU(x) = x * (0.5 + f1(t) + f2(f1(t)))，Horner 在 float64 下求值。"""
    t = x.to(_POLY_EVAL_DTYPE) / cfg["C"]
    f1 = horner_poly_jit(f1_f64, t)
    y = f1 + horner_poly_jit(f2_f64, f1)
    return (x.to(_POLY_EVAL_DTYPE) * (0.5 + y)).to(x.dtype)


def _single_gelu_from_t(
    x: torch.Tensor, cfg: dict, f_f64: torch.Tensor
) -> torch.Tensor:
    """single: GELU(x) = x * (0.5 + f(t))，f(t) 近似 y(t)，Horner 在 float64 下求值。"""
    t = x.to(_POLY_EVAL_DTYPE) / cfg["C"]
    y = horner_poly_jit(f_f64, t)
    return (x.to(_POLY_EVAL_DTYPE) * (0.5 + y)).to(x.dtype)


def gelu_eval_from_config(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    composite: GELU(x) = x * (0.5 + f1(t) + f2(f1(t))), t = x/C
    single:    GELU(x) = x * (0.5 + f(t)), f(t) 近似 y(t)
    """
    device = x.device
    if cfg["kind"] == "composite":
        f1_f64 = _coeffs_to_f64(cfg["f1_coeffs"], device)
        f2_f64 = _coeffs_to_f64(cfg["f2_coeffs"], device)
        return _composite_gelu_from_t(x, cfg, f1_f64, f2_f64)
    f_f64 = _coeffs_to_f64(cfg["f_coeffs"], device)
    return _single_gelu_from_t(x, cfg, f_f64)


class GeluPolyEvaluator:
    """单层 GeLU 多项式；系数在首次 forward 时缓存到 x.device。"""

    __slots__ = ("layer_idx", "level", "_cfg", "_cached_key", "_f", "_f1", "_f2")

    def __init__(self, layer_idx: int, level: int):
        self.layer_idx = layer_idx
        self.level = level
        self._cfg = gelu_config_for_layer(layer_idx, level)
        self._cached_key: tuple[torch.device, torch.dtype] | None = None
        self._f: torch.Tensor | None = None
        self._f1: torch.Tensor | None = None
        self._f2: torch.Tensor | None = None

    def _ensure_coeff_tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        key = (device, dtype)
        if self._cached_key == key:
            return
        # 系数固定缓存为 float64（临时精度方案，见 _POLY_EVAL_DTYPE）
        if self._cfg["kind"] == "composite":
            self._f1 = _coeffs_to_f64(self._cfg["f1_coeffs"], device)
            self._f2 = _coeffs_to_f64(self._cfg["f2_coeffs"], device)
            self._f = None
        else:
            self._f = _coeffs_to_f64(self._cfg["f_coeffs"], device)
            self._f1 = None
            self._f2 = None
        self._cached_key = key

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_coeff_tensors(x.device, x.dtype)
        if self._cfg["kind"] == "composite":
            return _composite_gelu_from_t(x, self._cfg, self._f1, self._f2)
        return _single_gelu_from_t(x, self._cfg, self._f)


def poly_gelu_low(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    return gelu_eval_from_config(x, gelu_config_for_layer(layer_idx, 0))


def poly_gelu_mid(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    return gelu_eval_from_config(x, gelu_config_for_layer(layer_idx, 1))


def poly_gelu_high(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    return gelu_eval_from_config(x, gelu_config_for_layer(layer_idx, 2))


GELU_POLY_FUNCS: list = [
    poly_gelu_low,
    poly_gelu_mid,
    poly_gelu_high,
]
