# ULAPM-SIS

ULAPM-SIS is a unified model for socially appropriate robot response planning from user utterances. Given a user utterance, the model jointly predicts affective and interaction-related signals, including emotion, response behavior, interpersonal distance, and SIS-related social state variables, and combines learned representations with constraint-aware planning to produce socially appropriate robot actions.

This repository contains the core implementation of the model together with the main training and evaluation entry points used in the project.

## Highlights

- Unified prediction of emotion, response behavior, interpersonal distance, and SIS-related social state.
- Constraint-aware planning for socially appropriate robot behavior-distance decisions.
- Training and evaluation code for offline multi-task metrics, protocol-based robot benchmarking, and human-reference evaluation.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers
- NumPy
- scikit-learn
- tqdm

## Installation

```bash
pip install -r requirements.txt
```

## Example Entry Points

### Training

```bash
python src/train_stage1_with_val_log.py \
  --npz <dataset.npz> \
  --split <split.npz> \
  --out_dir runs/full
```

### Evaluation

```bash
# Table I-style offline evaluation
python src/eval_offline_table1.py \
  --npz <dataset.npz> \
  --split <split.npz> \
  --ckpt <checkpoint.pt>

# Table II-style robot protocol evaluation
python src/eval_robot_protocol.py \
  --protocol <protocol.json> \
  --method full \
  --ckpt <checkpoint.pt>

# Human-reference evaluation
python src/eval_human_adjudicated_420.py \
  --relabel_csv <relabel.csv>
```

For the protocol benchmark, `src/eval_robot_protocol.py` supports `--method b1` (rule-based), `--method b2` (plain multi-task), `--method b3` (LLM prompting), and `--method full` (ULAPM-SIS / ablation checkpoints). Post-hoc clipping variants can be reproduced with `--method full --full_output_mode posthoc`.

## Repository Structure

- `src/models/`: ULAPM-SIS and baseline model definitions
- `src/train_stage1_with_val_log.py`: ULAPM-SIS training
- `src/train_b2_plain.py`: plain multi-task baseline training
- `src/train_stage1_ablate_nosis.py`: NoSIS ablation training
- `src/train_stage1_ablate_nohard.py`: NoHard ablation training
- `src/eval_offline_table1.py`: offline evaluation
- `src/eval_robot_protocol.py`: protocol evaluation
- `src/eval_human_adjudicated_420.py`: human-reference evaluation

## Note

This public release contains code only. Datasets, annotations, human-study assets, and model checkpoints are not included. Some training and evaluation scripts expect these files separately.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{wang2026ulapm,
  title={ULAPM-SIS: Unified Latent Affective Planning Model with Social Interaction State for Text-Driven Robot Behavior-Distance Decisions},
  author={Wang, Xu and Wang, Yiwei and Fang, Zheng and Togo, Shunta and Yokoi, Hiroshi and Jiang, Yinlai},
  year={2026},
  note={Manuscript under review}
}
```
