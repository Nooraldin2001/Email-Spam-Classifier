# Email Spam Classifier Project

This project aims to develop a machine learning model to classify emails as spam or ham (non-spam). The dataset used for training and testing the model is a collection of emails labeled as spam or ham.

## Project Overview

The project follows the typical data science pipeline:

1. **Data Acquisition**: The dataset is loaded from a CSV file containing email messages and their corresponding labels.
2. **Data Cleaning**: The dataset is cleaned by removing unnecessary columns and handling missing or duplicate values.
3. **Exploratory Data Analysis (EDA)**: Basic statistics and visualizations are used to explore the dataset and understand its characteristics.
4. **Text Preprocessing**: The text data in the emails is preprocessed by converting to lowercase, tokenization, removing special characters, stopwords, and punctuation, and stemming.
5. **Model Building**: Several machine learning models, including Naive Bayes, Support Vector Machines, Decision Trees, Random Forest, etc., are trained and evaluated for their performance in classifying spam and ham emails.
6. **Model Evaluation**: The performance of each model is evaluated using accuracy and precision metrics.
7. **Model Improvement**: Techniques like TF-IDF vectorization, ensemble methods (Voting Classifier, Stacking), and parameter tuning are applied to improve model performance.
8. **Deployment**: The best-performing model is saved for deployment in a production environment.

## Files

- **Project.ipynb**: Jupyter Notebook containing the project code.
- **vectorizer.pkl**: Pickle file containing the trained TF-IDF vectorizer.
- **model.pkl**: Pickle file containing the trained Naive Bayes classifier.

## Usage

1. Ensure you have Python installed on your system.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook `Project.ipynb` to train, evaluate, and save the model.
4. Use the saved vectorizer and model files (`vectorizer.pkl` and `model.pkl`) for classifying new email messages.

## Dependencies

- Python 3.x
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, nltk, wordcloud

## Acknowledgments

This is a project for the Bharat internship tasks 1st task is Email Spam Classifier.

