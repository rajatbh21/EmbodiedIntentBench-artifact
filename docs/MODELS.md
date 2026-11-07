# Perception models

The bundled **dynamic Lake study** uses three pretrained models. Each can be selected for local or
simulated remote execution, giving six routing actions. Remote execution changes
the service, transfer, and energy accounting; it is not a different checkpoint.
All three models receive RGB images and produce a binary person mask. Instance
masks are combined before the shared evaluator; ground truth never enters inference.

| Model | Pretraining and output | Configuration in the dynamic Lake study |
| --- | --- | --- |
| Mask R-CNN, ResNet-50 FPN v2 | TorchVision `COCO_V1`; instance segmentation | Resize to 1280 x 960; person score >= 0.5; mask probability >= 0.5 |
| YOLO11s-seg | Ultralytics COCO checkpoint; instance segmentation | `imgsz=960`, rectangular preprocessing, native-resolution masks, person score >= 0.5, NMS IoU 0.7; one unrotated view |
| CABiNet, MobileNetV3-Small | UAVid checkpoint; semantic segmentation | Resize to 1280 x 960; UAVid normalization; Human probability >= 0.5; one scale, no test-time augmentation |

The paper's **multi-phase Lake and Depot studies** instead use nine inference
configurations from four checkpoint variants: Mask R-CNN, YOLO11s, YOLO11n, and
CABiNet. The configurations vary full-frame/tiled processing, rotation, gamma
adjustment, and mask union; they are not nine independently trained models.
All execute locally in those studies. The [Depot figure sources](../examples/paper_figures/README.md)
retain their numeric outputs, but this artifact's model-fetch and fresh-inference
commands cover only the three dynamic Lake checkpoints listed above.

The benchmark's measured model times and assumed power values are separate from
the upstream accuracy/speed tables. New hardware can change execution time, and
reproducing routing results from frozen predictions does not measure new hardware.

## Fetch the exact weights

Run from the repository root with Python 3.11 or later:

```bash
python tools/fetch_models.py --list
python tools/fetch_models.py
python tools/fetch_models.py --verify-only
```

The first command displays every upstream URL, SHA-256, and byte size. Fetching
requires only the Python standard library. It downloads approximately 220 MiB
into ignored `.cache/` directories, verifies each complete file before installing
it atomically, and reuses already verified files. An interrupted download does
not become a model checkpoint. A mismatched existing file causes failure and is
preserved for inspection. The tool does not import source, unpickle checkpoints,
install packages, or run inference.

Use `--model maskrcnn`, `--model yolo11s`, or `--model cabinet` to select a family;
the option may be repeated. `--cache-root /path/to/cache` changes the destination.
`--verify-only` never downloads files or creates the generated CABiNet manifest.
Weights are retrieved from their publishers rather than committed to the paper
repository; all three checkpoints exceed 8 MB.

| Cache path, relative to `.cache/` | Pinned origin |
| --- | --- |
| `torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` | PyTorch model URL; full checksum in the fetch tool |
| `models/yolo11s-seg.pt` | Ultralytics assets release `v8.3.0`; full checksum in the fetch tool |
| `models/cabinet_uavid_small/` | CABiNet code commit `f48a36294000e7cc186efea07379b915be05418f`; Hugging Face model revision `25e46666435fd59d2033e0c47cc089cad5cf7da5` |

The CABiNet directory includes its original code license, model card, class
metadata, normalization source, configuration, and checkpoint. Its generated
`manifest.json` matches the original experiment manifest byte for byte:
`defe6058340cf7ae414c08758ab591fe930afc68b1730b6a1d62e8981a978ace`.
All twelve upstream CABiNet files have independent size and checksum pins.

## Optional native inference dependencies

Frozen routing replay does not require PyTorch, model weights, or Ultralytics.
For fresh model execution, install the optional dependencies:

```bash
python -m pip install '.[native]'
export TORCH_HOME="$PWD/.cache/torch"
```

Use a compatible PyTorch/TorchVision pair for the host, following the
[official installation instructions](https://pytorch.org/get-started/locally/).
The original CPU validation used Python 3.12, PyTorch 2.14.1+cpu,
TorchVision 0.29.1+cpu, Ultralytics 8.3.221, NumPy 2.3.5, Pillow 12.3.0,
and OpenCV headless 4.11.0.86. The dependency ranges in `pyproject.toml` are not
a complete environment lock; different versions require revalidation.

For an executor configuration loaded from disk, resolve the YOLO `checkpoint`
and CABiNet `bundle_manifest` to the above cache paths on the current machine.
Keep their checksum fields unchanged. `TORCH_HOME` makes TorchVision use the
verified Mask R-CNN file. CABiNet's executor checks the manifest and source before
importing its pinned Python implementation and uses a weights-only Torch loader.
Ultralytics loads its own checkpoint format through its upstream runtime.

## Check fresh predictions against the released evidence

```bash
python tools/check_native.py --output results/native-validation.json
```

This executes all three models on capture IDs 0, 9, and 17 of the bundled
`shoreline_loop/sector_3` recording, then compares the nine binary masks with
the stored predictions pixel for pixel. It verifies model files, the replay
bundle, and source hashes, and records the package versions in the output JSON.
It supplies only RGB to the models. Ground truth is not an inference input.
The tool adjusts model paths from `data/native_models.json` to the selected
cache root, so no original author's filesystem is required.

Use `--case` and `--captures` for a different declared subset. Each run requires
a new output filename. This check performs fresh neural inference, but reuses
the recorded flight; it does not launch Gazebo or remeasure the published service
latency profiles. Its observed host durations are diagnostic only.

## Third-party licenses and provenance

The benchmark's MIT license does not replace model or dependency licenses.

| Component | Upstream terms |
| --- | --- |
| TorchVision implementation | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) |
| Mask R-CNN checkpoint | [Official `COCO_V1` model](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html); TorchVision states that pretrained model terms can depend on the training dataset. Do not treat the code license as a separate blanket weights license. |
| Ultralytics implementation and YOLO11 models | [AGPL-3.0 or the publisher's Enterprise license](https://docs.ultralytics.com/models/yolo11/#citations-and-acknowledgments) |
| CABiNet source bundle | Apache-2.0, in the pinned upstream `LICENSE` downloaded with the source |
| CABiNet UAVid checkpoint | AGPL-3.0, declared by the pinned [model card](https://huggingface.co/dronefreak/cabinet-mobilenetv3-small-uavid/tree/25e46666435fd59d2033e0c47cc089cad5cf7da5) |

The [TorchVision pretrained-model notice](https://github.com/pytorch/vision#pre-trained-model-license)
describes the distinction between library and model terms. Preserve upstream
licenses and attribution when redistributing third-party components.
