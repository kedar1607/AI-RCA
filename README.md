# AI-RCA: Automated Root Cause Analysis

This project implements an automated Root Cause Analysis (RCA) system that uses machine learning to predict the root causes of Jira tickets based on log data. The system employs multiple similarity metrics including Cosine Similarity, Word-Wrap, and Levenshtein distance to make accurate predictions.

## Prerequisites

- Python 3.9.7
- Jupyter Notebook
- Required Python packages (will be listed in requirements.txt)

## Installation

1. Clone the repository using GitHub CLI:
```bash
gh repo clone kedar1607/AI-RCA
```

Or using Git:
```bash
git clone https://github.com/kedar1607/AI-RCA.git
cd AI-RCA
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage Instructions

1. **Generate Training Data**:
   - Open `dummy_data_gen_with_added_bugs.ipynb`
   - Adjust the parameters:
     - `NUM_SESSIONS_FOR_TRAINING`
     - `NUM_SESSIONS_CROSS_VALIDATION`
   - Run the Python cell to generate:
     - `analytics_logs.csv`
     - `datadog_mobile_logs.csv`
     - `backend_logs.csv`
     - Training and cross-validation Jira tickets

2. **Training and Inference**:
   - Open `training_inference.ipynb`
   - Run the first and second Python cells
   - Wait for the training and inference process to complete
   - Review the predictions in the output

## How It Works

The system uses a combination of similarity metrics to predict RCAs:
- Cosine Similarity: Measures the cosine of the angle between two vectors
- Word-Wrap: Analyzes text similarity
- Levenshtein Distance: Measures the minimum number of single-character edits required to change one string into another

## Important Notes

- Be careful when modifying the data generation logic as the generated bugs/Jira tickets need to be consistent with the logs
- The system requires a significant amount of computational resources for training
- Make sure to use Python 3.9.7 as specified, as other versions may cause compatibility issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

https://github.com/kedar1607/AI-RCA/blob/main/LICENSE

## Contact

For any questions or issues, please open an issue in the GitHub repository. 