# AI-RCA: Automated Root Cause Analysis

This project implements an automated Root Cause Analysis (RCA) system that uses machine learning to predict the root causes of Jira tickets based on log data. The system employs multiple similarity metrics including Cosine Similarity, Word-Wrap, and Levenshtein distance to make accurate predictions.

## Similarity Metrics

The system uses three different similarity metrics to analyze and compare text data:

1. **Cosine Similarity**
   - Measures the cosine of the angle between two vectors in an inner product space
   - Range: [-1, 1] where 1 indicates identical vectors
   - Implementation: scikit-learn's `cosine_similarity`
   - Documentation: [scikit-learn Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
   - Wikipedia: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

2. **Word-Wrap Similarity**
   - Analyzes text similarity by comparing word sequences and their arrangements
   - Useful for detecting similar error patterns in logs
   - Implementation: Custom implementation using text processing techniques
   - Python Documentation: [String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods)

3. **Levenshtein Distance**
   - Measures the minimum number of single-character edits required to change one string into another
   - Range: [0, ∞) where 0 indicates identical strings
   - Implementation: `python-Levenshtein` package
   - Documentation: [python-Levenshtein](https://rawgit.com/ztane/python-Levenshtein/master/docs/Levenshtein.html#Levenshtein-distance)
   - Wikipedia: [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance)

## Project Context

The log data used in this project is generated from an `imaginary` news app for iOS and Android that consolidates news articles (text, audio, and video) from different news sources. The app includes the following `imaginary` functionality and screens:

1. Splash screen
2. Login Screen with Error scenarios:
   - Login using email
   - Google authentication
   - Twitter authentication
3. Home screen
4. Article Screen with Error scenarios
5. Video Article Screen with error scenarios
6. Audio Article Screen with error scenarios
7. Share Screen
8. Search feature

The app sends logs to multiple systems:
- DataDog (`datadog_mobile_logs.csv`)
- Analytics system (`analytics_logs.csv`)
- Backend component (`backend_logs.csv`)

## Prerequisites

- Python 3.9.7
- Jupyter Notebook
- Required Python packages (listed in requirements.txt)

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

4. Install Jupyter Notebook if you haven't already:
```bash
pip install jupyter
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