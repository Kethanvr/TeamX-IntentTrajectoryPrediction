# TeamX — Intent and Trajectory Prediction

This repository is a Python research prototype for pedestrian and cyclist trajectory prediction. It contains a spatio-temporal graph convolutional model, training and evaluation scripts, and dataset utilities developed for the MAHE Mobility Challenge 2025 AI track.

The training script defaults to 8 observed frames and 12 prediction frames. These settings describe the experiment configuration; they are not a real-time vehicle safety guarantee or a production deployment claim.

## Repository layout

- `model.py` defines the ST-GCNN model.
- `train.py` contains the training loop and command-line options.
- `test.py`, `metrics.py`, and `utils.py` provide evaluation and data helpers.
- `datasets/` is the expected location for prepared dataset files.
- `checkpoint/` stores model checkpoints created by training.

## Data requirement

The training script expects prepared training and validation data under `datasets/<dataset>/train/` and `datasets/<dataset>/val/`. The nuScenes dataset and any conversion step are not provided as a turnkey download here. Obtain and prepare data under its terms before running an experiment.

## Run

After installing compatible PyTorch and the Python dependencies used by the scripts, place the prepared dataset in the expected directories and run:

    python train.py --dataset nuscenes

Training options, including sequence lengths and output tags, are available in `train.py`. Dependency versions are not pinned in a repository requirements file.
