from dataclasses import dataclass, fields, field, MISSING
from typing import Optional, get_origin, get_args
import argparse


@dataclass
class CompressionConfig:
    model: str = "facebook/opt-6.7b"
    device: int = 0
    factorize_src_model: str = ""
    nystrom_src_model: str = ""
    tokenizer_src: str = "mistralai/Mixtral-8x7B-v0.1"
    output_dir: str = "compressed_output"

    temp_storage_dir: str = "./compressed_output/layers/"

    dataset: str = "wikitext"

    nystrom_ridge: float = 1e-2

    order: Optional[str] = None

    calib_size: int = 32
    calibs_batch_size: int = 4

    compression_ratio: float = 0.5
    note: str = "NA"

    max_sparsity: float = 0.8
    sparsity_smoothing: float = 0.15

    ridge_vo: float = 1e-4
    ridge_qk: float = 1e-6

    debug: bool = False

    _parser_extras: dict = field(default=None, init=False, repr=False, compare=False)

    _FIELD_HELP = {
        "order": "mlp,qk,vo  -- <method>,<method>,<method>",
    }

    @classmethod
    def _resolve_type(cls, tp):
        """Unwrap Optional[X] → X, leave primitives unchanged."""
        origin = get_origin(tp)
        if origin is type(None):
            return None
        if origin is not None:  # e.g. Optional = Union[X, None]
            inner = [a for a in get_args(tp) if a is not type(None)]
            return inner[0] if inner else str
        return tp

    @classmethod
    def make_parser(cls, parser=None):
        parser = parser or argparse.ArgumentParser()
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            flag = f"--{f.name}"
            resolved = cls._resolve_type(f.type)
            if resolved is bool:
                parser.add_argument(flag, action="store_true", default=f.default)
            else:
                kwargs = {"type": resolved}
                if f.default is not MISSING:
                    kwargs["default"] = f.default
                else:
                    kwargs["required"] = True
                if f.name in cls._FIELD_HELP:
                    kwargs["help"] = cls._FIELD_HELP[f.name]
                parser.add_argument(flag, **kwargs)
        return parser

    @classmethod
    def from_args(cls, args=None):
        parser = cls.make_parser()
        parsed = parser.parse_args(args)
        init_fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in vars(parsed).items() if k in init_fields})

    def get(self, key: str, default=None):
        """Allow config.get('ridge_Cx', 1e-4) like a dict."""
        val = getattr(self, key, default)
        return val if val is not None else default

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key: str):
        return hasattr(self, key)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (useful for metrics)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
