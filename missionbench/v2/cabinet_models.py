"""Optional pinned local CABiNet UAVid backend; never downloads or trains at runtime.

The upstream source bundle is trusted executable code, verified against a pinned
manifest before import. Weights use Torch's restricted weights-only loader and
strict state matching. No source/weights are vendored into the benchmark package.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from missionbench.schemas.loading import SchemaValidationError
from missionbench.v2.real_models import BackendInfo, TorchSemanticSegmentationExecutor

_CACHE = {}
_REQUIRED = {
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/cabinet.py",
    "src/models/cab.py",
    "src/models/constants.py",
    "src/models/mobilenetv3.py",
    "configs/UAVid_info.json",
    "src/datasets/uavid.py",
    "cabinet_best.pth",
    "LICENSE",
    "README.md",
    "config.yaml",
}


def validate_parameters(params):
    if (
        params.get("model_id") != "cabinet_mobilenetv3_small_uavid"
        or params.get("device") != "cpu"
        or params.get("dtype") != "float32"
        or params.get("task_class") != "person"
    ):
        raise SchemaValidationError("CABiNet supports UAVid small, person, CPU float32 only")
    for key in ("bundle_manifest", "bundle_sha256"):
        if not isinstance(params.get(key), str) or not params[key]:
            raise SchemaValidationError(f"CABiNet requires {key}")
    if not Path(params["bundle_manifest"]).is_absolute():
        raise SchemaValidationError("CABiNet bundle_manifest must be absolute")
    warmup = params.get("warmup_runs", 1)
    if type(warmup) is not int or warmup < 0:
        raise SchemaValidationError("warmup_runs must be a nonnegative integer")


def verify_bundle(path: Path, expected: str):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("CABiNet manifest checksum mismatch")
    data = json.loads(path.read_text())
    if data.get("schema_version") != "0.1" or set(data["files"]) != _REQUIRED:
        raise ValueError("unsupported CABiNet bundle")
    root = path.parent.resolve()
    for name, sha in data["files"].items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest() != sha:
            raise ValueError(f"CABiNet file checksum mismatch: {name}")
    if (
        data.get("weights") != "cabinet_best.pth"
        or data.get("class_metadata") != "configs/UAVid_info.json"
    ):
        raise ValueError("unsupported CABiNet file mapping")
    return data


class CabinetBackend:
    def __init__(self, params):
        validate_parameters(params)
        manifest = Path(params["bundle_manifest"])
        data = verify_bundle(manifest, params["bundle_sha256"])
        root = manifest.parent.resolve()
        # The upstream module uses absolute src.models imports. Never shadow another
        # already-imported src package, or silently reuse code from a different bundle.
        for name, module in tuple(sys.modules.items()):
            if name == "src" or name.startswith("src.models"):
                file = getattr(module, "__file__", None)
                if file is None or not Path(file).resolve().is_relative_to(root):
                    raise ValueError("CABiNet src namespace conflicts with another package")
        import torch

        self.torch = torch
        start = time.perf_counter()
        sys.path.insert(0, str(root))
        try:
            source = importlib.import_module("src.models.cabinet")
            backbone = importlib.import_module("src.models.mobilenetv3")
        finally:
            sys.path.remove(str(root))
        classes = json.loads((root / data["class_metadata"]).read_text())
        self.target = next(c["trainId"] for c in classes if c["name"] == "Human")
        cfgs = backbone.mobilenetv3_small(num_classes=len(classes)).cfgs
        self.model = source.CABiNet(n_classes=len(classes), mode="small", cfgs=cfgs)
        weights = torch.load(root / data["weights"], map_location="cpu", weights_only=True)
        state = weights.get("model_state", weights)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        normalization = data["normalization"]
        self.mean = torch.tensor(normalization["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(normalization["std"]).view(1, 3, 1, 1)
        self.info = BackendInfo(
            framework="pytorch/CABiNet",
            framework_version=torch.__version__,
            model_id=params["model_id"],
            weights_id=params["weights_id"],
            weights_identifier=data["files"][data["weights"]],
            target_class="person",
            target_class_index=self.target,
            category_count=len(classes),
            device_requested="cpu",
            device_used="cpu",
            dtype_requested="float32",
            dtype_used="float32",
            input_width_px=params["input_width_px"],
            input_height_px=params["input_height_px"],
            preprocessing_id="RGB /255; UAVid mean/std; softmax Human; single scale, no TTA",
            model_load_s=time.perf_counter() - start,
            warmup_runs=params.get("warmup_runs", 1),
        )
        import numpy as np

        for _ in range(self.info.warmup_runs):
            self.infer(
                np.zeros((self.info.input_height_px, self.info.input_width_px, 3), dtype=np.uint8)
            )

    def infer(self, rgb):
        torch = self.torch
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        image = (image - self.mean) / self.std
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(image)[0]
            probabilities = logits.softmax(dim=1)[0, self.target].cpu().numpy()
        return probabilities, time.perf_counter() - start


def backend_for(params):
    # Verify on each construction, including when reusing a loaded model.
    verify_bundle(Path(params["bundle_manifest"]), params["bundle_sha256"])
    key = tuple(sorted(params.items()))
    if key not in _CACHE:
        _CACHE[key] = CabinetBackend(params)
    return _CACHE[key]


class TorchCabinetSegmentationExecutor(TorchSemanticSegmentationExecutor):
    def __init__(self, spec, *, backend=None):
        validate_parameters(spec.parameters)
        super().__init__(spec, backend=backend, backend_factory=backend_for)

    def run(self, rgb):
        result = super().run(rgb)
        return replace(
            result,
            diagnostics={
                **result.diagnostics,
                "executor_kind": "torch_cabinet_segmentation",
                "bundle_sha256": self._spec.parameters["bundle_sha256"],
                "training_domain": "UAVid oblique aerial; no guaranteed Gazebo/nadir performance",
            },
        )
