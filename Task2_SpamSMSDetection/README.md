# Task 2: Spam SMS Detection

## 🎯 Objective
To build a robust AI model capable of classifying SMS messages as either "spam" or "legitimate" (ham). This project leverages Natural Language Processing (NLP) techniques to analyze message content and identify patterns associated with spam.

## 📊 Dataset
This project uses the **SMS Spam Collection Data Set** from the UCI Machine Learning Repository, sourced via Kaggle.

- **Source:** [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Content:** The dataset contains 5,572 SMS messages in English, tagged with their respective labels (`ham` or `spam`).
- **Features:**
  - `label`: The target variable (spam/ham).
  - `message`: The raw text content of the SMS.

## ⚙️ Methodology

1.  **Data Preprocessing:**
    -   The raw text is cleaned by converting it to lowercase, removing all punctuation, and filtering out common English "stopwords" (e.g., 'the', 'a', 'is').
2.  **Feature Extraction:**
    -   The cleaned text messages are converted into a numerical format using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique. This method evaluates how relevant a word is to a document in a collection of documents.
3.  **Model Training:**
    -   Three different classification models were trained and evaluated:
        -   **Multinomial Naive Bayes:** A classic probabilistic algorithm that is highly effective for text classification.
        -   **Logistic Regression:** A robust linear model used as a strong baseline.
        -   **Support Vector Machine (SVM):** A powerful model that works well in high-dimensional spaces like text data.

## 🏆 Results
The models were evaluated based on their accuracy, precision, recall, and F1-score. The **Naive Bayes** classifier demonstrated the best overall performance, particularly in correctly identifying spam without misclassifying legitimate messages.

| Model                  | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
|------------------------|----------|------------------|---------------|-----------------|
| **Naive Bayes**        | 0.9830   | 1.00             | 0.87          | 0.93            |
| **Logistic Regression**| 0.9704   | 0.98             | 0.79          | 0.87            |
| **Support Vector Machine**| 0.9785 | 0.99             | 0.84          | 0.91            |

*Note: Results may vary slightly upon different runs.*

The best model (`Naive Bayes`) and the TF-IDF vectorizer have been saved as `spam_classifier_model.pkl` and `tfidf_vectorizer.pkl` for real-world predictions.

## 🚀 How to Run

1.  **Clone the Repository:**
    ```bash
    # This assumes you have already cloned the main CODSOFT repo
    cd CODSOFT/
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd Task2_SpamSMSDetection/
    ```

3.  **Install/Update Dependencies:**
    Make sure you have a virtual environment activated and run:
    ```bash
    pip install -r ../requirements.txt  # Assuming requirements.txt is in the root
    ```

4.  **Run the Script:**
    ```bash
    python spam_classifier.py
    ```
    The script will train the models, save the EDA plots in the `plots/` folder, and save the final model and vectorizer in the root of the task folder.