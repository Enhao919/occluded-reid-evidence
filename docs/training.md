# Training TIERNet on Occluded-Duke

Use the environment and dataset layout in the [README](../README.md).

## Inputs

| Input | Description |
| --- | --- |
| `DUKE` | Occluded-Duke root with its training, query, and gallery directories |
| `SWIN_B` | Swin-B checkpoint from [SOLIDER](https://github.com/tinyvision/SOLIDER#models) |
| `HEATMAPS` | Occluded-Duke training heatmaps from the [BPBreID parsing labels](https://github.com/VlSomers/bpbreid#download-human-parsing-labels) |
| `PART_LABELS` | Destination for the converted eight-part parsing cache |
| `SOURCE_SUPERVISION` | External shared-source supervision directory containing `cohort.json` and `weak_points.json.gz` for the reference training split |

The shared-source files describe training image pairs and their annotated points. The [evaluation protocol](../foundation/configs/occluded_duke.json) records the data and checkpoint fingerprints.

## Prepare the inputs

```bash
export DUKE=/path/to/Occluded-Duke
export SWIN_B=/path/to/solider_swin_base.pth
export HEATMAPS=/path/to/pifpaf_maskrcnn_filtering
export PART_LABELS=/path/to/part_labels
export SOURCE_SUPERVISION=/path/to/shared_source_annotations

python reproduce.py import-labels \
  --data-root "$DUKE" --heatmaps "$HEATMAPS" --part-labels "$PART_LABELS"

python reproduce.py check-data \
  --data-root "$DUKE" --part-labels "$PART_LABELS" \
  --pretrained "$SWIN_B" --source-supervision "$SOURCE_SUPERVISION"
```

## Train and evaluate

```bash
python reproduce.py train \
  --data-root "$DUKE" --part-labels "$PART_LABELS" \
  --pretrained "$SWIN_B" --source-supervision "$SOURCE_SUPERVISION" \
  --work-dir runs/occluded_duke --gpu 0

python reproduce.py test --data-root "$DUKE" \
  --work-dir runs/occluded_duke --output runs/occluded_duke_eval --gpu 0
```

The driver trains and refines the encoder, calibrates foreground batch normalization on the training split, and then fits the evidence branch with the encoder frozen. To continue training, run `reproduce.py resume` with the same options and work directory.
