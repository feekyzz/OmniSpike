# YOLOE-26 SNN Direct Text-Route SER

This repository is a clean reviewer-facing copy of the YOLOE-26 SNN instance-segmentation implementation centered on the S-scale configuration:

`ultralytics/cfg/models/26/omnispike-text-route-bira-reg16.yaml`

The original workspace is not modified by this package.

## Model scope

The retained model path contains:

- SNN backbone and neck blocks implemented in `ultralytics/nn/modules/yolo_spikformer.py`;
- the YOLOE SNN detection/segmentation heads in `ultralytics/nn/modules/head.py`;
- text-conditioned bi-level routing in `ultralytics/nn/modules/text_guided_ser.py`;
- the model parser and module registration in `ultralytics/nn/tasks.py` and `ultralytics/nn/modules/__init__.py`;
- the target S-scale configuration with `reg_max: 16`, `end2end: true`, and `text_model: mobileclip2:b`.

The P4 and P5 features are directly refined by the text-route module and then shared by the classification, box-regression, and mask-coefficient branches. The Proto branch reads the original P3 feature.

## Repository hygiene

This copy contains source code and lightweight configuration only. Local datasets, checkpoints, generated experiment outputs, caches, and unrelated experimental branches are intentionally excluded. The repository is therefore suitable for a fresh GitHub upload.

The MobileCLIP TorchScript weight is not committed. The existing Ultralytics download mechanism can retrieve `mobileclip2_b.ts` when text encoding is first used.

## Environment

Use Python 3.10 or 3.11 with a PyTorch build matching the available CPU/CUDA environment.

```powershell
python -m pip install -e .
python -m pip install -r requirements-reviewer.txt
```

The main package dependencies are defined in `pyproject.toml`; `requirements-reviewer.txt` lists the additional SNN and text-encoding dependencies used by this model path.

## Verify model construction

The following command constructs the exact reviewer configuration without training or writing experiment results:

```powershell
python verify_0824_direct_text_route_ser.py
```

Expected checks include:

- the final head is `omnispike_SNNSegment22_2DirectTextRouteSER`;
- `reg_max == 16`;
- `end2end == True`;
- both prediction branches and the Proto output are present.

A small-data training smoke script is also retained at:

`train-segment-snn2-direct-text-route-ser-smoke.py`

It is intended only for a local smoke test after the reviewer supplies the dataset and selects an output directory.

## Main files

| Purpose | Path |
|---|---|
| Target model configuration | `ultralytics/cfg/models/26/omnispike-text-route-bira-reg16.yaml` |
| SNN model head | `ultralytics/nn/modules/head.py` |
| Text-route SER | `ultralytics/nn/modules/text_guided_ser.py` |
| SNN backbone/neck blocks | `ultralytics/nn/modules/yolo_spikformer.py` |
| YAML parser and module registration | `ultralytics/nn/tasks.py` |
| Smoke training entry point | `train-segment-snn2-direct-text-route-ser-smoke.py` |
| Structural tests | `tests/test_direct_text_route_ser_spikes.py` |

## Reproducibility notes

The data YAMLs in this repository are small smoke-test configurations. Formal COCO/LVIS reproduction requires the reviewer’s own licensed/downloaded datasets and the corresponding checkpoint or training setup. Paths are intentionally not hard-coded to the original local machine.

This project retains the upstream Ultralytics license and notices. Please review the upstream license terms before redistribution.

