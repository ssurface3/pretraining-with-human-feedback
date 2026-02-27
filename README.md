# Pretraining Language Models with Human Preferences

This is a modernized fork of the codebase accompanying the paper [Pretraining Language Models with Human Preferences](https://arxiv.org/abs/2302.08582) (Korbak et al., 2023). It has been updated to work with **Python 3.12**, modern PyTorch/Transformers, and uses **Comet ML** for experiment tracking (replacing the original Weights & Biases integration).

The codebase is built around Hugging Face Transformers' `Trainer` and contains implementations of five objectives for pretraining with human feedback (PHF). PHF objectives annotate training data with rewards and overwrite `Trainer.compute_loss` to use them as additional training signal. Rewards are provided by `apo.scorers.Scorer`: an object that determines whether a piece of text is aligned or misaligned with human preferences (e.g. non-toxicity).

## Requirements

- **Python** 3.12+
- **PyTorch** 2.1+
- **Transformers** 4.36+
- **CUDA GPU** (tested on Tesla T4)
- **Comet ML** 3.30+ (for experiment tracking)

## Quickstart

```bash
pip install -r requirements.txt
python train.py --task configs/toxicity/pretrain.yml --method configs/toxicity/conditional.yml
```

### Quick test run (100 steps)

```bash
cd /kaggle/working/pretraining-with-human-feedback
CUDA_VISIBLE_DEVICES=0 python train.py \
  --task configs/toxicity/pretrain.yml \
  --method configs/toxicity/conditional.yml \
  --override training.max_steps=100 training.eval_steps=50
```

### Longer training (1000 steps)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --task configs/toxicity/pretrain.yml \
  --method configs/toxicity/conditional.yml \
  --override training.max_steps=1000 training.eval_steps=500
```

## Configuration

The `train.py` script requires two config files: **task** and **method**.

- **Task configs** (`configs/{task}/pretrain.yml` or `configs/{task}/finetune.yml`) define the dataset, tokenizer, model, and training hyperparameters.
- **Method configs** (`configs/{task}/{method}.yml`) define the objective and method-specific settings (e.g. conditional training prefixes, filtering thresholds).

### Key training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.max_steps` | Derived from `num_tokens` | Total optimizer steps |
| `training.eval_steps` | — | Evaluate every N steps |
| `training.per_device_train_batch_size` | 8 | Micro-batch size per GPU |
| `training.effective_batch_size` | 64 | Target effective batch size (gradient accumulation computed automatically) |
| `training.learning_rate` | 0.0005 | Peak learning rate |
| `training.fp16` | true | Mixed precision training |
| `training.warmup_ratio` | 0.01 | Fraction of steps for LR warmup |
| `training.weight_decay` | 0.1 | Weight decay |
| `training.seed` | 42 | Random seed |
| `generation.run_on_train_start` | true | Run generation evaluation before training |
| `generation.run_on_train_end` | true | Run generation evaluation after training |

### Overriding parameters

Any config value can be overridden from the command line:

```bash
python train.py \
  --task configs/toxicity/pretrain.yml \
  --method configs/toxicity/conditional.yml \
  --override training.max_steps=500 training.learning_rate=0.0001 training.per_device_train_batch_size=4
```

## Tasks

| Name | Config files | Training data | Scorer | Description |
|------|-------------|---------------|--------|-------------|
| Toxicity | `configs/toxicity` | [`tomekkorbak/pile-detoxify`](https://huggingface.co/datasets/tomekkorbak/pile-detoxify) | `DetoxifyToxicityScorer` | Toxicity probability via [detoxify](https://github.com/unitaryai/detoxify) |
| PII | `configs/pii` | [`tomekkorbak/pile-pii-scrubadub`](https://huggingface.co/datasets/tomekkorbak/pile-pii-scrubadub) | `PIIScorer` | PII count per character via [scrubadub](https://github.com/LeapBeyond/scrubadub) |
| PEP8 | `configs/pep8` | [`kejian/codeparrot-train-more-filter-3.3b-cleaned`](https://huggingface.co/datasets/kejian/codeparrot-train-more-filter-3.3b-cleaned) | `PEP8Scorer` | PEP8 violations per character via [pycodestyle](https://github.com/PyCQA/pycodestyle) |

## Objectives

| Name | Objective class | Description |
|------|----------------|-------------|
| MLE | `MLE` | Standard cross-entropy loss |
| Filtering | `MLE` | Requires `dataset.filter_threshold` in config |
| Conditional training | `MLE` | Requires `dataset.conditional_training_config` in config |
| Unlikelihood | `Unlikelihood` | Requires `objective.score_threshold` and `objective.alpha` |
| AWR | `AWR` | Requires `objective.alpha` and `objective.beta` |
| RWR | `AWR` | Special case of AWR with `objective.alpha=1` |

## Experiment Tracking (Comet ML)

This fork uses [Comet ML](https://www.comet.com/) in **offline mode** by default. Experiment logs are saved to `./comet_logs/`. To upload results:

```bash
comet upload ./comet_logs/<experiment_id>.zip
```

To use online mode, set your API key:
```bash
export COMET_API_KEY='your-key-here'
```

## Metrics

On each evaluation step, `apo.callbacks.GenerateAndScoreCallback` generates samples and computes:

- `score` — average misalignment score across generated samples
- `score_max@25` — average maximum score in 25 samples (similar to expected maximum toxicity in [RealToxicityPrompts](https://arxiv.org/abs/2009.11462))
- `current_samples` — table of samples with prompts and scores (logged to Comet ML)

`apo.callbacks.KLGPT3Callback` estimates KL divergence from GPT-3 (requires `OPENAI_API_KEY`; disabled gracefully if not set).

## Pretrained models

The models pretrained in the original paper are available on HuggingFace Hub:

| Objective | Toxicity | PEP8 | PII |
|-----------|----------|------|-----|
| MLE | [goofy_pasteur](https://huggingface.co/tomekkorbak/goofy_pasteur) | [mighty-mle](https://huggingface.co/kejian/mighty-mle) | [nervous_wozniak](https://huggingface.co/tomekkorbak/nervous_wozniak) |
| Filtering | [amazing_shannon](https://huggingface.co/tomekkorbak/amazing_shannon) | [mighty-filtering](https://huggingface.co/kejian/mighty-filtering) | [cocky_carson](https://huggingface.co/tomekkorbak/cocky_carson) |
| Conditional | [hungry_saha](https://huggingface.co/tomekkorbak/hungry_saha) | [mighty-conditional](https://huggingface.co/kejian/mighty-conditional) | [boring_mcclintock](https://huggingface.co/tomekkorbak/boring_mcclintock) |
| UL | [nifty_banach](https://huggingface.co/tomekkorbak/nifty_banach) | [mighty-ul](https://huggingface.co/kejian/mighty-ul) | [affectionate_wescoff](https://huggingface.co/tomekkorbak/affectionate_wescoff) |
| AWR | [upbeat_ramanujan](https://huggingface.co/tomekkorbak/upbeat_ramanujan) | [vigor-awr](https://huggingface.co/kejian/vigor-awr) | [confident_knuth](https://huggingface.co/tomekkorbak/confident_knuth) |
| RWR | [keen_clarke](https://huggingface.co/tomekkorbak/keen_clarke) | [mighty-rwr](https://huggingface.co/kejian/mighty-rwr) | [gifted_hugle](https://huggingface.co/tomekkorbak/gifted_hugle) |

## Codebase structure

```
.
├── train.py                 # Main training script
├── requirements.txt         # Python 3.12-compatible dependencies
├── apo/
│   ├── callbacks.py         # Evaluation pipeline (Comet ML logging)
│   ├── dataset_wrappers.py  # Iterable streaming blocks of tokens
│   ├── kl_gpt3.py           # KL divergence from GPT-3
│   ├── metrics.py           # Metrics on LM samples
│   ├── models.py            # GPT2LMHeadModel with value heads
│   ├── objectives.py        # Loss functions (MLE, AWR, Unlikelihood)
│   ├── scorer_utils.py      # Scorer utilities
│   ├── scorers.py           # Scoring LM samples and dataset elements
│   ├── trainer.py           # Custom HuggingFace Trainer subclass
│   └── utils.py             # Utility functions
├── configs/
│   ├── toxicity/            # Toxicity task configs
│   ├── pii/                 # PII task configs
│   └── pep8/                # PEP8 task configs
├── scripts/
│   └── dataset_builders/    # Dataset generation scripts
└── resources/               # Prompt lists, word lists
```

## Changes from the original repo

- **Python 3.12 compatibility** — updated all dependencies and fixed API breaking changes
- **Comet ML** replaces Weights & Biases for experiment tracking
- **Modern library support** — works with PyTorch 2.1+, Transformers 4.36+, Datasets 2.14+
- **Bug fixes** — `max_steps` override preserved correctly, DDP guard for single-GPU, graceful fallbacks for optional callbacks

See `changes_in_the_code.txt` for a detailed changelog of all 33 code modifications.

## Citing

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.08582,
  doi = {10.48550/ARXIV.2302.08582},
  url = {https://arxiv.org/abs/2302.08582},
  author = {Korbak, Tomasz and Shi, Kejian and Chen, Angelica and Bhalerao, Rasika and Buckley, Christopher L. and Phang, Jason and Bowman, Samuel R. and Perez, Ethan},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Pretraining Language Models with Human Preferences},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
