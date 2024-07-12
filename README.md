# Poleval 2024: Emotion and Sentiment Recognition

This repository contains the code and models developed for the Poleval 2024 Emotion and Sentiment Recognition Challenge. The goal of the challenge is to accurately recognize emotions and sentiments from a given dataset of text reviews.

## Challenge Details
The challenge is organized by Poleval, and more details can be found on the [official competition page](https://beta.poleval.pl/challenge/2024-emotion-recognition).

### Training Dataset
The provided training dataset consists of 6393 sentences extracted from 776 reviews. Each sentence is labeled with one or more of the following 11 categories:

- **Plutchik's wheel of emotions**: Joy, Trust, Anticipation, Surprise, Fear, Sadness. Disgust and Anger
- **Perceived Sentiment**: Positive, Negative and Neutral

### Test Datasets
Two test datasets have been provided, each containing 167 reviews with 1234 and 1264 sentences respectively. These test datasets do not include the multi-class classification labels, which are the targets for the model predictions.

## (My) Model Overview
The model developed for this challenge is an ensemble of five well-known machine learning models. These models are trained on numerical representations of the text reviews, which are transformed into 76 features. These features are scaled to the range [0, 1] and include both discrete and continuous variables.

### Feature Extraction
The feature extraction process involves four different versions of the original reviews, with each version contributing 19 features. These features are obtained using:

1. **LSTM Model**: A Long Short-Term Memory (LSTM) network, denoted as `lstm` in the code, is used to process the text and generate features.
2. **Pretrained Models from Hugging Face**: Three different pretrained models from the Hugging Face library, collectively referred to as `hfam` (Hugging Face All Models) in the code, are utilized to extract meaningful features from the text:
   - **Herbert**: Uses the model `dkleczek/Polish-Hate-Speech-Detection-Herbert-Large`, which is fine-tuned for Polish language tasks, particularly hate speech detection.
   - **XLM-RoBERTa**: Employs the model `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`, designed for sentiment analysis in multiple languages, making it well-suited for multilingual sentiment classification.
   - **Multilingual BERT**: Utilizes the model `nlptown/bert-base-multilingual-uncased-sentiment`, a version of BERT trained for sentiment analysis across various languages, providing robust features for sentiment recognition.

### Ensemble Approach
The ensemble model combines the predictions of the following five models:

1. **Random Forest**: An ensemble method using decision trees.
2. **XGBoost**: An efficient implementation of gradient boosting.
3. **MLP (Multi-Layer Perceptron)**: A neural network with multiple layers.
4. **CNN (Convolutional Neural Network)**: A neural network specialized for processing grid-like data, such as text sequences.
5. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.

By leveraging the strengths of these diverse models, the ensemble aims to achieve high accuracy in recognizing both emotions and sentiments from the text data.

The ensemble model works by querying each of the five models (after appropriate hyperparameter tuning) to make predictions. For each emotion or sentiment category, the class that is selected by the majority of models (at least three out of five) is chosen as the final prediction. This majority voting mechanism helps to improve the robustness and accuracy of the predictions by combining the different strengths and perspectives of the individual models.

## Data
The `data` directory contains three subdirectories: `testA`, `testB`, and `train`. Each subdirectory includes four files corresponding to four different data formats, as well as a file with numerical observations for the ensemble model.

### Subdirectory Structure
Each subdirectory (`testA`, `testB`, and `train`) contains the following files:

1. **in_baseline**: This file is the original dataset `X` provided by the challenge organizers. It contains a single column labeled "text", with each row containing a clean review written in Polish.
2. **in_gpt_corr**: This file is the corrected version of `in_baseline`, produced using Chat GPT-3.5 Turbo. Each line of the original text was independently corrected with the command "Correct the following text to proper Polish." The total cost for nearly 10,000 corrections using GPT was less than a dollar.
3. **in_prep_bas**: This file is the `in_baseline` dataset after applying preprocessing steps following Exploratory Data Analysis (EDA). The preprocessing involved:
   - Converting the text to lowercase.
   - Removing non-word characters using regular expressions.
   - Stripping extra spaces to clean up the text.
4. **in_prep_gpt**: This file is the `in_gpt_corr` dataset after applying the same preprocessing steps as `in_prep_bas`, i.e., converting text to lowercase, removing non-word characters, and stripping extra spaces.

These four versions of the dataset ensure that the models are trained on diverse representations of the text data, enhancing their robustness.

### Numerical Observations
Each subdirectory also contains a file named `concated_for_ensemble_final.csv`. This file includes 76 numerical observations per review, representing the transformed text data. These numerical features are in the range [0, 1] and include both discrete and continuous variables. They are derived from four versions of the original reviews, with each version contributing 19 features. These features are obtained using:
- **LSTM Model**: Denoted as `lstm` in the code, a Long Short-Term Memory network is used to process the text and generate features.
- **Pretrained Models from Hugging Face**: Collectively referred to as `hfam` (Hugging Face All Models) in the code, three different pretrained models are used to extract meaningful features from the text:
  - **Herbert**: `dkleczek/Polish-Hate-Speech-Detection-Herbert-Large`,
  - **XLM-RoBERTa**: `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`,
  - **Multilingual BERT**: `nlptown/bert-base-multilingual-uncased-sentiment`.

### Training Labels
The `train/` subdirectory also contains a file named `expected.tsv`. This file represents the target variable `y` and contains the binary classification labels for the training data. Each review is labeled with 11 binary values corresponding to the following categories:
- Emotions from Plutchik's wheel of emotions: joy, trust, anticipation, surprise, fear, sadness, disgust, anger.
- Perceived sentiment: positive, negative, neutral.

## Ensemble Model Construction
The main script for building the final ensemble model is located in `src/models/ensemble_final.ipynb`. This script orchestrates the training and evaluation of multiple machine learning models and combines their predictions to create a robust ensemble model for emotion and sentiment recognition.

### Data Loading and Preparation
The script loads the preprocessed training data from `data/train/concated_for_ensemble_final.csv` and the corresponding labels from `data/train/expected.tsv`. The data is then split into training and testing sets to evaluate model performance during training.

### Model Definitions and Hyperparameter Tuning
The script defines several machine learning models:
1. **Random Forest**
2. **XGBoost**
3. **MLP (Multi-Layer Perceptron)**
4. **CNN (Convolutional Neural Network)**
5. **Naive Bayes**

For the first three models (Random Forest, XGBoost, MLP), hyperparameter tuning is performed using GridSearchCV to find the best model parameters. The CNN model is defined separately, and the Naive Bayes model is trained independently for each label without hyperparameter tuning.

### Model Training
Each model is trained on the training data:
- **Random Forest, XGBoost, MLP**: These models are trained using a multi-label classification approach with the tuned hyperparameters.
- **CNN**: The CNN model is trained on the training data transformed to the appropriate input shape.
- **Naive Bayes**: Separate Naive Bayes models are trained for each label, and their performance is evaluated individually.

### Achieved Accuracies
The accuracies achieved by each model are as follows:

| Model           | Accuracy   |
|-----------------|------------|
| Random Forest   | 0.9205     |
| XGBoost         | 0.9149     |
| MLP             | 0.9149     |
| CNN             | 0.8919     |
| Naive Bayes     | 0.9041     |

### Model Evaluation
After training, the models are evaluated on the test set. Accuracy scores are calculated to assess the performance of each model. These scores are printed for each model to give a quick overview of their effectiveness.

### Super Ensemble Model
To combine the strengths of the individual models, a super ensemble model is created using a majority voting mechanism. For each prediction, the ensemble model queries all five trained models. The final prediction for each label is determined by the majority vote (i.e., the class chosen by at least three of the five models).

### Saving Models and Making Predictions
The trained models and the super ensemble model function are saved to the `models/ensemble_final/` directory. The script also loads test datasets (`data/testA/concated_for_ensemble_final.csv` and `data/testB/concated_for_ensemble_final.csv`) and makes predictions using the super ensemble model. The predictions are saved to `predictions/testA/ensemble_final/predictions.csv` and `predictions/testB/ensemble_final/predictions.csv`.

### Additional Script and Logging with Weights & Biases
In the same directory, there is an additional script named `ensemble_final_with_wandb.ipynb`. This script extends the main script by integrating logging with the Weights & Biases tool. A dedicated subfolder, `wandb`, has also been provided, containing the generated outputs with the help of this tool.

## Input Data Preparation Details
Everything is based on the same earlier mentioned four variants of the data: `baseline`, `gpt_corr`, `prep_bas`, and `prep_gpt`. For each of these variants, and for all three datasets (`train`, `test-A`, and `test-B`), 19 numerical features are generated:

- **11 features**: These are predicted categories based on an LSTM model. The architecture of the LSTM model is as follows:
  - An embedding layer with an input dimension of 5000 and an output dimension of 128.
  - An LSTM layer with 64 units.
  - A dense layer with 11 units and a sigmoid activation function.

- **2 features**: These come from the `dkleczek/Polish-Hate-Speech-Detection-Herbert-Large` model, which provides a binary label and a score between 0 and 1.

- **4 features**: These are obtained from the `XLM-RoBERTa` model, which include:
  - Sentiment categories: negative, neutral, and positive (one-hot encoded into three columns).
  - A score between 0 and 1.

- **2 features**: These come from the `Multilingual BERT` model, which provides:
  - A label ranging from 1 star to 5 stars (mapped to values 0, 0.25, 0.5, 0.75, and 1).
  - A score between 0 and 1.

For training, all 76 features (4 variants * 19 features each) are used. Despite some repetitions and dependencies among the features, it was deemed beneficial to feed the models with larger amounts of data to maintain a reasonable number of dimensions.

By combining the different features extracted from these models, the ensemble approach leverages diverse aspects of the text data, contributing to more robust and accurate predictions.

## src/
The `src/` directory contains all the source code for data preprocessing, model training, and evaluation. It is organized into two subdirectories: `data_preprocessing` and `models`.

### data_preprocessing/
This subdirectory contains scripts for data exploration, preprocessing, and generating corrected versions of the datasets.

- **baseline_eda.ipynb** and **gpt_corr_eda.ipynb**: These notebooks perform Exploratory Data Analysis (EDA) for two of the four data variants, specifically for the baseline and the GPT-3.5 Turbo corrected data variants.
- **generate_and_save_preprocessed_data.ipynb**: This notebook handles preprocessing and saving the corrected data. The preprocessing steps include converting the text to lowercase, removing non-word characters, and stripping extra spaces.
- **in_(testA)|(testB)|(train)_gpt_corr.ipynb**: These scripts generate the `_gpt_corr` versions of the files by replacing the original text with the corrected text using the prompt "Correct the following text to proper Polish."
- **concat_predictions_for_(testA)|(testB)|(train).ipynb**: These scripts concatenate all predictions from the LSTM and Hugging Face models, resulting in the `concated_for_ensemble_final.csv` tables used for the ensemble model.

### models/
This subdirectory contains scripts for training and evaluating the machine learning models, including the ensemble model.

- **ensemble_final.ipynb**: The main script for creating the final ensemble model. It includes training, hyperparameter tuning, and evaluation of multiple models. An HTML version of this notebook is also available.
- **cross_validation/**: This subdirectory contains four scripts for performing 5-fold cross-validation for the LSTM model across the four data variants.
- **train_save/**: Contains scripts for training on the entire training dataset and saving the models (for LSTM) and predictions (for Hugging Face models). There is also a somewhat experimental script, `phsd_baseline_train.ipynb`, which focuses on training a single Hugging Face model (`dkleczek/Polish-Hate-Speech-Detection-Herbert-Large`).
- **load_predict/**: Contains scripts for loading the saved LSTM models and making predictions on the test datasets.

This organization ensures that the workflow is well-structured and that each step in the data processing and model training pipeline is reproducible and easy to follow.

## Directory and File Structure
In addition to the main scripts and data directories, the repository contains several other important directories and files.

### models/
This directory contains the saved models and tokenizers used in the project.

- **LSTM Models**: At the root of the `models/` directory, you will find the LSTM models for all four data variants (`baseline`, `gpt_corr`, `prep_bas`, `prep_gpt`). Each model is approximately 8 MB in size and is saved in the `.h5` format.

- **Tokenizers**: In the `models/tokenizers/` subdirectory, there are four tokenizer files in `.pickle` format, each corresponding to one of the data variants mentioned above.

- **Ensemble Models**: The `models/ensemble_final/` subdirectory contains the saved models from the final ensemble model script (`ensemble_final.ipynb`). Note that the Random Forest model is not included here due to its large size (75 MB), which exceeds GitHub's file size limits.

### plots/
The `plots/` directory contains visualizations generated during the Exploratory Data Analysis (EDA) phase.

- **Baseline and GPT-Corrected Data**: There are four plots each for the `baseline` and `gpt_corr` data variants. These plots show the distributions of word counts and review lengths (in characters) within the reviews.

### predictions/
This directory holds the prediction results for all three datasets (`train`, `test-A`, and `test-B`), each in its respective subdirectory.

- **LSTM Predictions**: Files named `lstm_(baseline)|(gpt_corr)|(prep_bas)|(prep_gpt)` contain the LSTM model predictions for the corresponding data variants.
- **Hugging Face Model Predictions**: Files named `hfam_(baseline)|(gpt_corr)|(prep_bas)|(prep_gpt)` contain the predictions from the Hugging Face models.
These prediction files are core inputs for the final models created in `ensemble_final.ipynb`.

Each of these subdirectories (`train`, `test-A`, `test-B`) also contains a file named `ensemble_final/predictions.csv`, which holds the final predicted values based on the ensemble model. The `predictions.csv` files for `test-A` and `test-B` are used for the competition submission.

- **Mean LSTM Accuracy Results**: In `predictions/train/`, there is a file named `mean_lstm_accuracy_results.xlsx`, which documents the performance metrics of the LSTM network across five iterations of 5-fold cross-validation. These metrics are fairly consistent.
