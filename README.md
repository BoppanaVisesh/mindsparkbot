# MindSpark

MindSpark is a child-friendly chatbot backend and fine-tuning setup.

## Install

```bash
pip install -r requirements.txt
```

## Training Data

The training script now combines:

- Local dataset in `datasets/custom_dataset.json`
- Hugging Face `empathetic_dialogues`
- Hugging Face `daily_dialog`
- Hugging Face `eli5`
- Hugging Face `go_emotions`

All data is normalized into:

- `instruction`
- `input`
- `response`

Default training uses about 2000 samples total:

- 400 custom
- 400 empathetic dialogues
- 400 daily dialogue turns
- 400 ELI5 samples
- 400 GoEmotions samples

## Train

```bash
python training/train.py
```

Default CPU-friendly settings:

- `num_train_epochs=1`
- `per_device_train_batch_size=8`
- `gradient_accumulation_steps=1`
- `learning_rate=2e-5`
- `logging_steps=50`
- `save_steps=500`
- `max_length=128`
- `dataloader_num_workers=2`
- `dataloader_pin_memory=False`

Fast mode (recommended first run):

```bash
python training/train.py --fast
```

Full mode without ELI5:

```bash
python training/train.py --skip-eli5
```

Custom epochs:

```bash
python training/train.py --fast --epochs 2
```

The fine-tuned model is saved to `models/mindspark_model`.
