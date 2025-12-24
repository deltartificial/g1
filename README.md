# G1

Simulate, train and evaluate Unitree G1 humanoid robot using Genesis physics engine on Apple Silicon (Metal).

## Setup

```bash
pip install -r walk/requirements.txt
```

## Usage

```bash
cd walk
make train    # Train
make eval     # Evaluate
```

### Training

```bash
PYTHONUNBUFFERED=1 python3 -u g1_train.py -B 4096 --max_iter 1000
```

| Parameter | Description |
|-----------|-------------|
| `PYTHONUNBUFFERED=1` | Disable output buffering for real-time logs |
| `-u` | Force unbuffered stdout/stderr |
| `-B` | Number of parallel environments |
| `--max_iter` | Maximum training iterations |
| `--device` | Device: `mps` (default), `cuda`, `cpu` |
| `--backend` | RL library: `auto`, `rsl_rl`, `sb3` |
