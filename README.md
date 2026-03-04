## UniSemAlign: Semi-supervised Gland Segmentation

### Environment and dependencies

- **Python**: 3.9+ recommended.
- **Create environment (optional)**:

```bash
conda create -n unisemalign python=3.9 -y
conda activate unisemalign
```

- **Install Python dependencies** (from inside the repo root):

```bash
pip install -r UniSemAlign/requirements.txt
```

### Installing CONCH (optional text encoder)

If you enable the text branch (`text_proto.text_encoder: "conch"`), you need CONCH and a HuggingFace access token.

- Follow the official CONCH repository for installation:
  - `https://github.com/Mahmoodlab/CONCH`
- In short, you will typically do something like:

```bash
pip install "git+https://github.com/Mahmoodlab/CONCH.git"
```

- Obtain a HuggingFace token (with access to `MahmoodLab/uni` and `MahmoodLab/conch`) and export it:

```bash
export HF_TOKEN="your_hf_token"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

The training script will also read `hf_token` from the YAML configs if you prefer to store it there.

### Data layout

The configs assume a relative data root:

- `data_root: dataset`
- Partitions live under:
  - `UniSemAlign/partitions/glas_10/`, `glas_20/`
  - `UniSemAlign/partitions/crag_10/`, `crag_20/`

Each partition folder contains text files (`labeled.txt`, `unlabeled.txt`, `val.txt`, `test.txt`) listing image IDs.
The raw images and masks are expected under `dataset/...` as used in the original CorrMatch code (e.g. `dataset/glas/10/...`).

If you need to regenerate partitions, use the partition tool in `UniSemAlign/tools` (see that file for examples).

### Training

Run all commands from the repository root (`Semi-MoE`).

- **Semi-supervised CRAG**:

```bash
cd UniSemAlign
bash tools/train_crag_20.sh   # CRAG 20%
bash tools/train_crag_10.sh   # CRAG 10%
```

- **Semi-supervised GLAS**:

```bash
cd UniSemAlign
bash tools/train_glas_10.sh   # GLAS 10%
bash tools/train_glas_20.sh   # GLAS 20%
```

Each script wraps a call to:

```bash
python UniSemAlign.py --config <config.yaml> \
  --labeled-id-path <path_to_labeled_txt> \
  --unlabeled-id-path <path_to_unlabeled_txt> \
  --save-path <experiment_output_dir>
```

You can also use `tools/train.sh` to launch training with `torchrun` if you want multi-GPU.

### Generating masks and evaluation scores

`UniSemAlign/train.txt` lists typical commands; the key scripts are:

- `tools/gen_masks_glas_10.sh`
- `tools/gen_masks_glas_20.sh`
- `tools/gen_masks_crag_10.sh`
- `tools/gen_masks_crag_20.sh`

From the repo root:

```bash
cd UniSemAlign
bash tools/gen_masks_crag_20.sh <checkpoint_path.pth> [split] [out_dir] [mode] [port]
```

Where:

- `split`: `val` or `test`
- `mode`: `score` | `mask` | `both` (default)
- `out_dir`: where masks (PNG) are saved if `mode` includes `mask`


- **Masks only** (no scores):

```bash
bash tools/gen_masks_crag_20.sh exp/crag/20/corrmatch/uni_70.845.pth test
```

- **Scores only**:

```bash
bash tools/gen_masks_glas_10.sh exp/glas/10/corrmatch/uni_dice4.pth test . score
```

- **Both masks and scores**:

```bash
bash tools/gen_masks_crag_10.sh exp/crag/10/corrmatch/uni_dice4.pth test visual/crag10
```

Adjust the checkpoint paths and output directories to match your experiments.

