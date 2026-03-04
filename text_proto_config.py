import copy
from typing import Any, Dict


def get_text_proto_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = copy.deepcopy(cfg.get("text_proto", {}) or {})

    def _bool_from(d: Dict[str, Any], name: str, default: bool) -> bool:
        v = d.get(name, default)
        if isinstance(v, str):
            v_low = v.strip().lower()
            if v_low in ("1", "true", "yes", "y", "on"):
                return True
            if v_low in ("0", "false", "no", "n", "off"):
                return False
        return bool(v)

    def _float_from(d: Dict[str, Any], name: str, default: float) -> float:
        v = d.get(name, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    def _int_from(d: Dict[str, Any], name: str, default: int) -> int:
        v = d.get(name, default)
        try:
            return int(v)
        except Exception:
            return int(default)

    norm = {
        "enable": True,
        "use_text": True,
        "use_learnable_proto": True,
        "lambda_pixel_proto_ce": 1.0,
        "lambda_pixel_text_ce": 1.0,
        "lambda_proto_text_align": 1.0,
        "gamma_deeplab_logits": 1.0,
        "alpha_proto_logits": 0.1,
        "beta_text_logits": 0.1,
    }


    text_encoder = str(raw.get("text_encoder", "") or "").lower()
    class_names = raw.get("class_names", None)
    if isinstance(class_names, (list, tuple)):
        class_names = [str(c) for c in class_names]
    elif isinstance(class_names, str):
        class_names = [s.strip() for s in class_names.split(",") if s.strip()]
    else:
        class_names = None

    norm["text_encoder"] = text_encoder
    if class_names is not None:
        norm["class_names"] = class_names

    coop_raw = copy.deepcopy(raw.get("coop", {}) or {})
    coop_cfg = {
        "n_ctx": _int_from(coop_raw, "n_ctx", 4),
        "learnable": _bool_from(coop_raw, "learnable", True),
    }
    norm["coop"] = coop_cfg

    return norm

