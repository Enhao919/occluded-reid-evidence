# Identity Evidence for Occluded Person Re-Identification

A SOLIDER Swin-B model for Occluded-Duke with shared-source supervision, three-state part evidence allocation, and evidence-constrained graph re-ranking.

[Method](#method) · [Results](#occluded-duke-results) · [Setup](#setup) · [Training](#training) · [Evaluation](#evaluation)

## Method

**Foundation.** Train a multi-scale, eight-part BPBreID head on SOLIDER Swin-B for 120 epochs. Refine the network with background interventions and a frozen identity teacher through R4 epoch 180. Calibrate the foreground batch-normalization statistics on the training split, then freeze the foundation.

1. **Shared-source identity supervision.** Match visual points shared by two target crops and compare each point with cross-camera identity references. The training targets are *target support*, *other support*, and *uncertain*.
2. **Three-state evidence allocation.** Train a 33,164-parameter head that adjusts the spatial pooling weights and visibility of the eight parts. The objective combines part GiLt, balanced point cross-entropy, and cosine preservation of the frozen part features.
3. **Evidence-constrained graph re-ranking.** Use the adjusted part embeddings for the initial distance and the predicted evidence quality for a node-wise KL budget. Refine support weights on a fixed k-reciprocal graph before computing the Jaccard distance.

The model takes RGB crops at inference. Training annotations are required for fitting the model. Full-gallery evaluation also uses the training images to calibrate the evidence-quality scale.

## Occluded-Duke results

All three rows use the same reference network weights. Evaluation used 2,210 queries and the full 17,661-image gallery with same-identity, same-camera matches removed at scoring. The reference run used BF16, batch size 256, and a single forward pass per image.

| Scoring method | Correct at Rank-1 | Rank-1 | mAP |
| --- | ---: | ---: | ---: |
| Foreground and visible parts | 1,806 | 81.7195% | 70.2107% |
| Fixed k-reciprocal re-ranking | 1,888 | 85.4299% | 84.4090% |
| Evidence-constrained graph re-ranking | 1,887 | **85.3846%** | **84.4222%** |

The reported scores come from the original checkpoints. A new training run produces a new set of weights and should be evaluated separately.

## Setup

The reference configuration uses Linux, Python 3.10, PyTorch 2.7.1, torchvision 0.22.1, and one BF16-capable GPU. The commands below use the CUDA 12.8 wheels.

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Keep external assets outside the repository. Set these paths for training.

| Variable | Required input |
| --- | --- |
| `DUKE` | Occluded-Duke directory with `bounding_box_train/`, `query/`, and `bounding_box_test/` |
| `SWIN_B` | SOLIDER/LUPerson Swin-B teacher checkpoint or its compact reference export |
| `HEATMAPS` | BPBreID training heatmaps under `pifpaf_maskrcnn_filtering/` |
| `PART_LABELS` | Destination of the converted eight-part parsing cache |
| `SOURCE_SUPERVISION` | External shared-source training annotations, with `cohort.json` and `weak_points.json.gz` from the reference experiment |

See the upstream [SOLIDER model instructions](https://github.com/tinyvision/SOLIDER#models) and [BPBreID parsing label instructions](https://github.com/VlSomers/bpbreid#download-human-parsing-labels). The accepted data and checkpoint fingerprints are in [the Occluded-Duke protocol](foundation/configs/occluded_duke.json).

The shared-source cohort identifies the two crops, their identities, and reference-image indices. The point file stores correspondence grids, part indices, support directions, and conflict indicators. These are training-only annotations. The pipeline consumes the reference cache, so training E requires access to those two files. The dataset manifests are derived from the external dataset at runtime.

### Convert and verify training inputs

```bash
export DUKE=/path/to/Occluded-Duke
export SWIN_B=/path/to/swin_base.pth
export HEATMAPS=/path/to/pifpaf_maskrcnn_filtering
export PART_LABELS=/path/to/bpbreid_8part_cache
export SOURCE_SUPERVISION=/path/to/shared_source_annotations

python reproduce.py import-labels \
  --data-root "$DUKE" --heatmaps "$HEATMAPS" --part-labels "$PART_LABELS"

python reproduce.py check-data \
  --data-root "$DUKE" --part-labels "$PART_LABELS" \
  --pretrained "$SWIN_B" --source-supervision "$SOURCE_SUPERVISION"
```

An existing compatible part cache can be passed directly. The data check verifies the sample order, image checksums, labels, and supplied pretraining file.

## Training

```bash
python reproduce.py train \
  --data-root "$DUKE" --part-labels "$PART_LABELS" \
  --pretrained "$SWIN_B" --source-supervision "$SOURCE_SUPERVISION" \
  --work-dir runs/occluded_duke --gpu 0
```

The driver runs P120, R4 refinement through epoch 180 with its 300-epoch learning-rate schedule, training-only foreground BN calibration, and six rounds of evidence training. The foundation batch size is 64 and the image resolution is 384 × 128.

Continue with the same inputs and work directory.

```bash
python reproduce.py resume \
  --data-root "$DUKE" --part-labels "$PART_LABELS" \
  --pretrained "$SWIN_B" --source-supervision "$SOURCE_SUPERVISION" \
  --work-dir runs/occluded_duke --gpu 0
```

## Evaluation

Evaluate a run produced by the training driver.

```bash
python reproduce.py test \
  --data-root "$DUKE" --work-dir runs/occluded_duke \
  --output runs/occluded_duke_eval --gpu 0 --batch-size 256
```

To replay the reference results with independently supplied checkpoints, pass the original R180, foreground BN, and E checkpoints together.

```bash
python reproduce.py test \
  --data-root "$DUKE" \
  --native-checkpoint /path/to/epoch180_preserved.pt \
  --bn-statistics /path/to/train_only_statistics.pt \
  --evidence-checkpoint /path/to/allocation_seed101.pt \
  --output runs/reference_eval --gpu 0 --batch-size 256
```

The evaluator saves all three scores, the per-query rankings, and the evidence calibration and graph diagnostics. Use a new output directory for each evaluation. Full-gallery re-ranking uses dense distance matrices and is intended for machines with sufficient host memory.

## Source layout

| Path | Contents |
| --- | --- |
| `reproduce.py` | Data preparation, training, resume, and evaluation commands |
| `foundation/` | P120 and R180 recipes, training code, and foreground calibration |
| `evidence/allocation.py` | Three-state part head, spatial pooling, and point supervision |
| `evidence/fit.py` | Evidence training, loss diagnostics, and checkpoints |
| `evidence/bounded_graph.py` | Evidence quality, support correction, and graph re-ranking |
| `evaluate_full.py` | Occluded-Duke full-gallery scoring |
| `tests/` | Training-lineage and workflow checks |

## Acknowledgments

The backbone is based on [SOLIDER](https://github.com/tinyvision/SOLIDER). The part head and training losses build on [BPBreID](https://github.com/VlSomers/bpbreid) and [KPR](https://github.com/VlSomers/keypoint_promptable_reidentification). The fixed re-ranking comparator follows [Zhong et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html). See [third-party notices](NOTICE.md) for source and license details.
