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

The original CSV is now hosted on Hugging Face:
https://huggingface.co/datasets/RobaireTH/Fraud-Detection-System-Dataset
