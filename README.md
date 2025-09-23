# text4rec: A Framework for Text-Enhanced Sequential Recommendation

This repository contains the code for `text4rec`, a framework for building and experimenting with sequential recommendation models that leverage text-based item embeddings.

## Architecture

The general architecture involves using pre-trained text embeddings to represent items, which are then fed into a sequential model to learn user preferences.

<img src=Images/archtecture.png width=800>

### Key Improvements

While B4R (BERT4Rec) achieves high accuracy by calculating a softmax loss over the entire item vocabulary, its performance is known to degrade when using a sampled softmax, often falling short of SASRec. This challenge is discussed in the paper at https://arxiv.org/abs/2308.07192.

To overcome this limitation, we have implemented a novel approach by replacing the B4R architecture with **SASRec** and using a **gBCE (Generalized Binary Cross-Entropy)** loss.

This method avoids the computationally expensive full softmax required by B4R, enabling efficient training on large-scale datasets while aiming to achieve comparable or superior accuracy.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Install Poetry:** Follow the instructions on the [official website](https://python-poetry.org/docs/#installation).

2.  **Install Dependencies:** From the root of the project directory, run:
    ```bash
    poetry install
    ```
    This will create a virtual environment and install all the necessary packages defined in `pyproject.toml`.

## Usage

### Datasets

The framework is designed to work with sequential datasets like [MovieLens](https://grouplens.org/datasets/movielens/) and [Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/). The dataloaders will automatically download the required dataset if it's not found locally. The dataset can be specified in the configuration files.

### Running Experiments

The main experiments can be reproduced by running the `experiment.sh` script.

```bash
./cmd/experiment.sh
```

This script runs a series of experiments defined within it. Each line calls `sandbox/run_and_log.sh` with a specific configuration file and hyperparameters.

The core configurations are located in the `params/` directory:
-   `b4r.yaml`: Configuration for the baseline BERT4Rec model.
-   `t4r.yaml`: Configuration for the text-enhanced BERT4Rec model (T4R).
-   `sasrec.yaml`: Configuration for our proposed gSASRec model.

You can modify `cmd/experiment.sh` to run different experiments or create new `.yaml` configuration files to test different models and hyperparameters.