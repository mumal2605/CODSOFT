# Customer Churn Prediction

## 🎯 Objective
The primary goal of this project is to develop a machine learning model that can predict customer churn for a telecommunications company. By analyzing historical customer data, the model identifies key factors contributing to churn and predicts which customers are at high risk of leaving. This allows the company to proactively engage with at-risk customers to improve retention.

## 📊 Dataset
This project uses the **Telco Customer Churn** dataset, sourced from Kaggle. It contains customer account information, demographic data, and services they have signed up for.

- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Features Include:** `gender`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`, `InternetService`, and more.
- **Target Variable:** `Churn` (Yes/No).

## 🚀 How to Run
To run this project, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/CODSOFT.git
    cd CODSOFT/Task1_CustomerChurn/
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Make sure you have the `requirements.txt` file in your directory.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**
    Launch Jupyter and open the `churn_model.ipynb` notebook.
    ```bash
    jupyter notebook churn_model.ipynb
    ```
    You can then run the cells sequentially to see the entire workflow from data loading to model evaluation and saving.

## 🏆 Example Result
The models were evaluated on Accuracy, Precision, Recall, F1-Score, and ROC-AUC. XGBoost was selected as the best-performing model due to its superior ROC-AUC score and excellent balance between precision and recall.

Here is a summary of the model performance on the test set:

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| **XGBoost**         | 0.793    | 0.524     | 0.757  | 0.620    | 0.831   |
| **Random Forest**   | 0.785    | 0.519     | 0.630  | 0.569    | 0.820   |
| **Logistic Regression**| 0.749    | 0.518     | 0.778  | 0.621    | 0.830   |

The best model (`XGBoost`) and the data preprocessor have been saved as `model.pkl` and `preprocessor.pkl` respectively for future use.