# DA6401_Assignment3

# DA6401_Assignment3: Character-Level Transliteration with Seq2Seq Models
==========================================================================
This repository contains the implementation and experimental results for an academic assignment (DA6401) focused on building and evaluating Sequence-to-Sequence (Seq2Seq) models for character-level transliteration. The project explores two distinct Seq2Seq architectures: a vanilla (non-attention) model and an enhanced attention-based model, comparing their performance in converting Latin script words to Devanagari script.


============================
## Project Overview
============================
The primary goal of this assignment is to understand, implement, and analyze the effectiveness of Seq2Seq models for a sequence-to-sequence mapping task, specifically transliteration. Transliteration involves converting text from one script to another, preserving the pronunciation rather than the meaning (unlike translation). This project uses the Hindi transliteration dataset, which maps Latin script words to their Devanagari equivalents.

The repository is organized into two main parts, each corresponding to a different model architecture:

* **`PartA/`**: Implements and evaluates the **Vanilla Seq2Seq Model**. This serves as a baseline to understand the fundamental capabilities of recurrent networks in sequence generation.
* **`PartB/`**: Implements and evaluates the **Attention-based Seq2Seq Model**. This advanced architecture is expected to perform better by allowing the decoder to dynamically weigh the importance of input characters during output generation.

All experiments are tracked and visualized using **Weights & Biases (W&B)**, providing comprehensive insights into training progress, hyperparameter tuning, and model performance.

=========================
## Dataset Information
==========================
The dataset used for this project is a subset of the **Dakshina dataset v1.0**, specifically the Hindi (`hi`) transliteration lexicon.

* **Source**: [Dakshina Dataset v1.0](https://storage.googleapis.com/sanskrit_data/dakshina_dataset_v1.0.tar) (The specific files used are expected to be available locally or in the Kaggle environment as per the paths in the code).
* **Format**: The data files are in TSV (Tab Separated Values) format. Each line typically contains a pair of words: `[target_script_word]\t[source_script_word]\t[optional_meta_info]`. In this project, the relevant columns are the Devanagari target and the Latin source.
* **Splits**: The dataset is provided in three splits:
    * `hi.translit.sampled.train.tsv`: For model training.
    * `hi.translit.sampled.dev.tsv`: For model validation and early stopping during training.
    * `hi.translit.sampled.test.tsv`: For final, unbiased evaluation of the trained models.
* **Preprocessing**: The `Vocabulary` class dynamically builds character-to-index mappings from the training data, including special tokens (`<sos>`, `<eos>`, `<pad>`, `<unk>`). The `TransliterationDataset` and `collate_fn` handle converting text sequences to numerical tensors and padding batches for efficient processing by recurrent networks.



#============================
## Repository Structure:    #
#============================

DA6401_Assignment3/
├── README.md                      # This main project overview.
├── PartA/
│   ├── README.md                  # Details about the Vanilla Seq2Seq Model.
│   ├── DL_Assignment4PartA.ipynb  # Jupyter Notebook for Vanilla Model.
│   └── predictions_vanilla/
│       └── Q5.csv                 # Test predictions from Vanilla Model.
└── PartB/
├── README.md                  # Details about the Attention-based Seq2Seq Model.
├── DL_Assignment4PartB.ipynb  # Jupyter Notebook for Attention Model.
└── prediction_attention/
└── Q6.csv                 # Test predictions from Attention Model.


=========================================
## Setup and Usage Guide
===========================================
To run the code and reproduce the experiments, follow these steps:

### Prerequisites

* Python 3.8+
* PyTorch
* Weights & Biases (`wandb`)
* NumPy
* tqdm
* scikit-learn
* matplotlib (for `PartB` heatmaps)
* Access to the Dakshina Hindi transliteration dataset. The code expects the data at `/kaggle/input/dakshina-dl-a3/dakshina_dataset_v1.0/hi/lexicons/` if running in a Kaggle environment, or adjust `BASE_DATA_DIR` in the notebooks accordingly.


==========================
### Installation
==========================
bash:
# Clone the repository
git clone [https://github.com/D4deben/DA6401_Assignment3.git](https://github.com/D4deben/DA6401_Assignment3.git)
cd DA6401_Assignment3

# Install Python packages
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118) # For CUDA 11.8; adjust as needed
pip install wandb numpy tqdm scikit-learn matplotlib jupyterlab




==============================
Weights & Biases Setup
==============================
This project heavily relies on Weights & Biases for experiment tracking.

Sign up for a free W&amp;B account at wandb.ai.
Log in to W&amp;B in your environment:
Bash

wandb login
You will be prompted to enter your API key. If running in Kaggle, the code attempts to retrieve it from Kaggle secrets.




=================================
Running the Experiments
Each part (PartA and PartB) contains a dedicated Jupyter notebook. You can run them sequentially.




1. Part A: Vanilla Seq2Seq Model
Navigate: cd PartA/
Open Notebook: Launch Jupyter Lab/Notebook and open DL_Assignment4PartA.ipynb.
Execution: Run all cells in the notebook.
The notebook will perform the following:
Load and preprocess the Hindi transliteration dataset.
Define and initialize the Vanilla Seq2Seq model.
Train the model using the BEST_HYPERPARAMETERS defined in the notebook (these should ideally be populated from your own hyperparameter tuning sweeps).
Evaluate the trained model's performance on the development and test sets.
Save the test predictions to PartA/predictions_vanilla/Q5.csv.
Log all training and evaluation metrics, and sample predictions to your W&amp;B project.


2. Part B: Attention-based Seq2Seq Model
Navigate: cd PartB/
Open Notebook: Launch Jupyter Lab/Notebook and open DL_Assignment4PartB.ipynb.
Execution: Run all cells in the notebook.
The notebook will perform the following:
Load and preprocess the Hindi transliteration dataset, ensuring vocabulary consistency.
Define and initialize the Attention-based Seq2Seq model.
Train the model using the BEST_ATTN_HYPERPARAMETERS defined in the notebook (again, populate with your optimal findings).
Evaluate the trained model's performance on the development and test sets.
Save the test predictions to PartB/prediction_attention/Q6.csv.
Log all training and evaluation metrics, sample predictions, and a character-level confusion matrix to your W&amp;B project.
Generate Attention Heatmaps: For a few selected test examples, it will generate and save attention heatmaps (PNG files) to PartB/attention_heatmaps_q5d/. These visualizations help understand what parts of the source sequence the model attends to while generating specific target characters.


After Running:
Check W&amp;B Dashboard: Visit your W&amp;B project dashboard (https://www.google.com/search?q=https://wandb.ai/YOUR_USERNAME/DL_A3) to explore the logged runs, compare model performances, analyze training curves, view sample predictions, and inspect confusion matrices.
Access Predictions: The generated prediction files (Q5.csv and Q6.csv) will be available in their respective predictions_vanilla/ and prediction_attention/ subdirectories. These files can be used for further analysis or submission.

Review Heatmaps: The generated attention heatmaps will be saved in PartB/attention_heatmaps_q5d/. You can embed these images directly into reports or presentations to illustrate the attention mechanism.

