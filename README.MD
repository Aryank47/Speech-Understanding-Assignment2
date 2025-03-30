# Q1: Speaker Verification & Speech Separation with SepFormer

## Overview

This code implements a comprehensive pipeline for speaker verification and speech separation/enhancement. It uses:

1. **ArcFace Loss** for fine-tuning a pre-trained speaker verification model.
2. **LoRA (Low-Rank Adaptation)** to efficiently fine-tune linear layers in the network.
3. **SepFormer** (from SpeechBrain) for speech separation of multi-speaker mixtures.
4. A novel pipeline that combines speech separation and speaker identification in one flow.

Key functionalities:

- **Pre-trained Speaker Verification Model** (using WavLM from Hugging Face).
- **LoRA** to replace `nn.Linear` layers for efficient adaptation.
- **ArcFace** to fine-tune speaker embeddings.
- **Multi-speaker mixing and dataset generation** (mimicking the style of LibriMix).
- **Baseline and pipeline evaluations** using EER, TAR, SDR, SIR, SAR, PESQ, and rank-1 accuracy.

## Requirements

- Python 3.7 or higher
- PyTorch (GPU recommended)
- torchaudio
- numpy
- scipy (some parts of the code might use `scipy` indirectly)
- scikit-learn (for EER, TAR using `roc_curve`)
- [mir_eval](https://github.com/craffel/mir_eval) (for SDR, SIR, SAR metrics)
- [pesq](https://pypi.org/project/pesq/) (for PESQ scores)
- [transformers](https://github.com/huggingface/transformers) (for WavLMModel)
- [speechbrain](https://github.com/speechbrain/speechbrain) (for the SepFormer model)

Install them, for example:

```bash
pip install torch torchaudio numpy scikit-learn mir_eval pesq transformers speechbrain
```

## File/Directory Structure

Below is a suggested directory structure. You can adapt as needed:

```
.
├── code1_speaker_verification_sepformer.py  # The Python code
├── vox1
│   └── trial_pairs.txt
│   └── wav/ ...
├── vox2
│   ├── aac/ ...
│   └── txt/ ...
└── pretrained_sepformer/  # Directory that SpeechBrain might create for storing the downloaded model
```

- **vox1** and **vox2** folders contain VoxCeleb1 and VoxCeleb2 data as required.
- **trial_pairs.txt** is used for speaker verification trials.

## How to Run

1. **Data Preparation**

   - Make sure you have VoxCeleb1 and VoxCeleb2 datasets downloaded and extracted, with paths matching what the code uses (or you can edit the paths in the code).
   - For VoxCeleb2, you should have both `aac` and `txt` subdirectories:
     - `aac/` with audio files
     - `txt/` with metadata references
   - For VoxCeleb1, you should have a similar `wav/` structure and a `trial_pairs.txt` specifying the enrollment and test pairs.

2. **Install Dependencies**  
   Ensure all packages listed in **Requirements** are installed in your environment.

3. **Run the Script**

   ```bash
   python code1_speaker_verification_sepformer.py
   ```

   The script will:

   - Load the pre-trained WavLM speaker model.
   - Evaluate its performance on VoxCeleb1 trial pairs (prints EER, TAR).
   - Fine-tune the model on a portion of VoxCeleb2 data (using LoRA + ArcFace).
   - Re-evaluate performance on VoxCeleb1.
   - Create a multi-speaker dataset from VoxCeleb2 (mixing utterances).
   - Evaluate baseline SepFormer for speech separation + speaker verification.
   - Train a new pipeline that combines separation + speaker verification.
   - Print out final metrics (SDR, SIR, SAR, PESQ, rank-1 accuracy).

4. **Interpreting Outputs**  
   The code prints:
   - **EER (Equal Error Rate)**, **TAR@FAR** for speaker verification.
   - **SDR, SIR, SAR, PESQ** for separation quality.
   - **Rank-1 Accuracy** for speaker identification among separated sources.

## Important Notes

- **GPU Usage**: If you have CUDA enabled, the code will detect and run on GPU. Otherwise, it may be slow on CPU.
- **Memory**: The code calls garbage collection and CUDA memory clearing frequently to handle large waveforms.
- **Parameter Tuning**: You can adjust hyperparameters like `r=4`, `alpha=1.0` in the LoRA function, or `margin=0.5`, `scale=64` in ArcFace, as well as `lr=1e-4`.

## Possible Extensions

- Use more advanced sampling or trimming strategies.
- Integrate additional loss functions for multi-task learning.
- Experiment with different pre-trained transformer-based audio models (e.g., Hubert, wav2vec2).

---

# Q2: Language Identification with MFCC and Feedforward NN

## Overview

This code trains a language identification (multi-class classification) model using MFCC features extracted from audio files in multiple Indian languages. Key steps include:

1. **MFCC Extraction** with `torchaudio`.
2. **Statistical Analysis** of MFCC distributions (mean, variance, heatmaps).
3. **Train/Validation/Test split** on extracted MFCC features.
4. **Feedforward Neural Network** for classification.
5. **Early Stopping** and **Confusion Matrix** visualization.

## Requirements

- Python 3.7 or higher
- PyTorch
- torchaudio
- numpy
- matplotlib
- scikit-learn (for confusion matrix)
- seaborn (for heatmap visualization)
- joblib (for caching/storing results, if desired)

Install them, for example:

```bash
pip install torch torchaudio numpy matplotlib scikit-learn seaborn joblib
```

## File/Directory Structure

```
.
├── code2_language_detection.py       # The Python script
├── Language Detection Dataset
│   ├── Bengali/
│   │   └── audio1.mp3
│   │   └── audio2.mp3
│   ├── Gujarati/
│   ├── Malayalam/
│   ├── Hindi/
│   ├── Marathi/
│   ├── Kannada/
│   ├── Punjabi/
│   ├── Tamil/
│   ├── Telugu/
│   └── Urdu/
└── <possible additional files>
```

- Each language folder contains `.mp3` audio files.

## How to Run

1. **Dataset Preparation**  
   Ensure your folder structure matches the code references:

   - `DATASET_PATH = './Language Detection Dataset'`
   - `LANGUAGES = ['Bengali','Gujarati','Malayalam','Hindi','Marathi','Kannada','Punjabi','Tamil','Telugu','Urdu']`
   - Each of these subfolders should contain `.mp3` files.

2. **Install Dependencies**  
   Make sure all needed libraries are installed.

3. **Run the Script**

   ```bash
   python code2_language_detection.py
   ```

   The script will:

   - Load and process all `.mp3` files from each language directory.
   - Extract MFCC features (with or without resampling).
   - Perform comparative analysis (mean, variance, heatmaps).
   - Prepare data for classification (train-validation-test split).
   - Train a simple feedforward neural network with an Early Stopping mechanism.
   - Plot the training and validation loss over epochs.
   - Evaluate on the test set, printing the final accuracy and confusion matrix.

4. **Interpreting Outputs**
   - **Loss Curves**: The training and validation loss curves will indicate if the model is converging.
   - **Confusion Matrix**: Gives insights into which languages are often misclassified.
   - **Accuracy**: Final test accuracy, printed at the end.

## Configuration and Hyperparameters

- **SAMPLE_RATE = 16000**: Resampling rate if `resample_waveform=True`.
- **N_MFCC = 40**: Number of MFCC coefficients.
- **BATCH_SIZE = 64**: Adjust based on GPU memory.
- **hidden_dim = 225**: Hidden layer dimension in the feedforward network.
- **learning_rate = 1e-4**: Training learning rate.
- **NUM_EPOCHS = 50**: Max training epochs (early stopping may stop earlier).

You can modify these in the script to tune performance.

## Potential Pitfalls

- Insufficient data for certain languages may cause imbalance in training.
- Audio files that fail to load or have corrupt audio frames will be skipped (the code prints warnings if so).
- If you have `.wav` files instead of `.mp3`, adapt the code or rename the extension checks.

## Extensions

- Use more sophisticated architectures (CNN/RNN/Transformers) instead of the feedforward network.
- Incorporate data augmentation (noise addition, random speed changes).
- Fine-tune a pre-trained model such as Wav2Vec2, HuBERT, or WavLM for language ID.
