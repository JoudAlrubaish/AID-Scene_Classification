# AID Scene Classification

This project develops and compares CNN models built from scratch for multi-class aerial scene classification using a selected subset of the AID dataset.

## Dataset

The project uses the following 10 AID scene classes:

- Forest
- Desert
- Beach
- River
- Mountain
- DenseResidential
- Industrial
- Airport
- Stadium
- Farmland

The raw images are not included in the repository because of their large size.

After downloading the AID dataset, place the selected class folders inside:

```text
data/raw/AID/
```

The expected folder structure is:

```text
data/raw/AID/
├── Airport/
├── Beach/
├── DenseResidential/
├── Desert/
├── Farmland/
├── Forest/
├── Industrial/
├── Mountain/
├── River/
└── Stadium/
```

The fixed training, validation, and test split files are included in `data/splits/`. These files ensure that all models use the same data partitions for a fair comparison.

## Models

The project includes the following CNN models:

- **Model A — Baseline CNN:** two convolutional blocks.
- **Model B — Deeper CNN:** three convolutional blocks.
- **Model C — Regularized CNN:** data augmentation, Batch Normalization, and Dropout.
- **Model D — Multi-Scaled CNN:** parallel 3×3 and 5×5 convolutional branches.
- **Model E — Custom Deep Light-Regularized CNN:** four convolutional blocks with lighter regularization.

Additional experiments using Early Stopping and learning-rate reduction are included in separate notebooks.

## Main Results

| Model | Test Accuracy | Macro F1 | Test Loss | Parameters | Epochs |
|---|---:|---:|---:|---:|---:|
| Model A | 63.64% | 0.62 | 1.03 | 24,202 | 15 |
| Model B | 68.37% | 0.66 | 0.88 | 102,154 | 15 |
| Model C | 52.84% | 0.53 | 1.56 | 102,826 | 15 |
| Model D | 68.18% | 0.67 | 0.84 | 118,538 | 15 |
| Model E | 75.76% | 0.76 | 0.67 | 422,602 | 15 |

Model E achieved the highest test accuracy under the shared 15-epoch experimental constraint and was selected for deployment.

The deployed model is saved as:

```text
artifacts/best_model.keras
```

## Project Structure

```text
AID-Scene_Classification/
├── artifacts/           # Saved CNN models
├── data/
│   ├── raw/AID/         # Raw images, excluded because of size
│   └── splits/          # Fixed train, validation, and test CSV files
├── notebooks/           # Model training and evaluation notebooks
├── results/
│   ├── figures/         # Training curves and confusion matrices
│   └── *.json           # Training histories and evaluation results
├── main.py              # FastAPI and Gradio application
├── project_utils.py     # Shared preprocessing and model utilities
├── pyproject.toml       # Project dependencies
└── README.md
```

## Installation

Install the required dependencies from the project directory:

```bash
uv sync
```

## Running the Notebooks

Start Jupyter Notebook using:

```bash
uv run jupyter notebook
```

The raw AID images must be available under `data/raw/AID/` to rerun data preparation or model training.

The trained models and saved results can still be used when the raw dataset is not included.

## Running the Application

Start the FastAPI and Gradio application using:

```bash
uv run uvicorn main:app --reload
```

After the server starts, open:

- FastAPI status: `http://127.0.0.1:8000/`
- API documentation: `http://127.0.0.1:8000/docs`
- Gradio interface: `http://127.0.0.1:8000/gradio`

The Gradio interface allows the user to upload an aerial image and displays the predicted scene class and confidence score.

## Prediction API

The `POST /predict` endpoint accepts an uploaded image and returns:

- `predicted_class`: the predicted AID scene category.
- `class_index`: the numerical index of the predicted class.
- `confidence`: the probability assigned to the predicted class.

## Saved Outputs

The project contains:

- Trained Keras models inside `artifacts/`.
- Training histories inside `results/`.
- Training and validation curves inside `results/figures/`.
- Confusion matrices for model evaluation.
- Classification reports inside the executed notebooks.

## Final Model

The Custom Deep Light-Regularized CNN combines the increased depth of Model B with lighter regularization than Model C.

The architecture uses:

- Four convolutional blocks with 32, 64, 128, and 256 filters.
- Data augmentation.
- A dropout rate of 0.2.
- A dense layer with 128 units.

The model achieved:

- Test accuracy: **75.76%**
- Macro F1-score: **0.76**
- Test loss: **0.6720**
- Parameters: **422,602**

The final model is loaded by `main.py` from `artifacts/best_model.keras` and is used for predictions through FastAPI and Gradio.