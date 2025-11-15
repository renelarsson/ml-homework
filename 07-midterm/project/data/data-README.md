# Data Directory

## Dataset Details
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File**: `creditcard.csv`
- **Size**: ~284,807 transactions with 492 fraud cases (~0.17% fraud rate).
- **Format**: CSV file.

## Instructions
### Using Kaggle CLI
1. Ensure the Kaggle CLI is installed and configured with your API key.
   - Install Kaggle CLI: `pip install kaggle`
   - Configure API key: Place `kaggle.json` in `~/.kaggle/` and set permissions: `chmod 600 ~/.kaggle/kaggle.json`
2. Download and unzip the dataset:
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p ../data/
   unzip -o ../data/creditcardfraud.zip -d ../data/
   ```
3. Verify the dataset file `creditcard.csv` exists in the `../data/` directory.

### Manual Download
1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Download the `creditcard.csv` file manually.
3. Place the file in this directory.
