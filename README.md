# Credit Card Fraud Detection

## Project Summary
This project presents a comprehensive analysis and machine learning model for fraud detection on a Kaggle dataset. It covers exploratory data analysis (EDA), data preprocessing, handling class imbalance, model training, optimization and evaluation.

## Dataset Summary
- **Source:** Kaggle Credit Card Fraud Detection dataset.
- **Size:** Approximately 284,807 transactions.
- **Features:** 31 total features.
- **Class Imbalance:** Fraudulent transactions represent a very small fraction (~0.17%) of all transactions, requiring specialized handling techniques such as SMOTE for balancing.

## Key Achievements
- Conducted thorough EDA to understand data distributions and feature relationships.
- Applied data cleaning and preprocessing for optimal model performance.
- Addressed class imbalance using SMOTE.
- Achieved a strong clasification metrics:

|   Metric  | Score  |
|-----------|--------|
| Precision | 99.85% |
| Recall    | 97.86% |
| F1-Score  | 98.84% |
| AUC-ROC   | 99.97% |
| Accuracy  | 98.85% |

## Deliverables
- Jupyter notebok with EDA and preprocessing steps.
- Model training and evaluation scripts.
- Visualizations supporting insights and model performance.
- A Streamlit UI for the model.

## How to Use

- Review the notebook for step-by-step analysis.
- Run the training script to reproduce model results.
- Use evaluations scripts to validate on new data.
- For the Streamlit UI, to get the evaluation metrics, you should upload a CSV file with the Class column intact. Though, you can still make predictions without it. Also, make sure you have dropped the Time column. The model wasn't fitted with it. 

## Viewing the Jupyter Notebook on Github?

If the Notebook doesn't render properly, consider using Binder or Google Colab by simply downloading the notebook by clicking "Raw" and saving the file. If not cloud execution, open the notebook locally using Jupyter Notebook or JupyterLab.

## Next Steps
Update Frequency: Not yet determined.
Want to make changes, or continue with the project? Here's what's next
- Feature importance analysis using SHAP or permutation importance.
- Deployment as an API
- ~Experiment with other classifiers~ and hyperparameter tuning.

## Quick Note
The model running in the Stremlit UI is the "model_xgboost.pkl". While testing the models with the same dataset, I discovered that the Random Forest classifer- "frauddetector.pkl" and the XGBRFClassifier- "model_xgbrf.pkl" have a significant drop in either Precision score or F1 score or both on the same dataset. Though, the Recall, Accuracy and AUC Scores are still pretty much high (95+ %). But the XGBClssifier model worked well, giving no sign of overfitting. It was able to generalize on the training data. This will be revisited and the threshold for the classifiers will be tuned to balance precision and recall. I had used a hard threshold throughout this work.

Please, when testing with a new dataset, make sure the dataset has the same distribution as the original dataset used in training this model. If you use a dataset with a low signal-to-noise ratio, the performance will be hurt. Also, if you use a data that does not preserve the relationship between the features, the model will get confuse trying to generalize.

# Data Files Removed

The original CSV files were removed to reduce project size. 

You can load them directly from Hugging FAce using: 
 ```python
 from datasets import load_dataset

dataset = load_dataset("RobaireTH/Fraud-Detection-System-Dataset", data_files={
    "original": "creditcard.csv",
    "refined": "refined_data.csv"
}) 
```

Then convert to pandas where needed using:
```python
original_data = dataset["original"].to_pandas()
refined_data = dataset["refined"].to_pandas()
```

The original CSV is now hostel on Hugging Face: 
https://huggingface.co/datasets/RobaireTH/Fraud-Detection-System-Dataset
## License

MIT License

```txt
Thank you for staying around,
Author: Mayowa Temitope, AKINYELE
```

# fraud-detection-system
