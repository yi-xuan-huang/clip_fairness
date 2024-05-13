import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import re

def extract_impression(text):

    impr_match = re.search(r'IMPRESSIONS?\s*:(.*)', text, re.IGNORECASE|re.DOTALL)
    out = f"IMPRESSION: {impr_match.group(1).strip()}" if impr_match else ""   
    return out

def main():
    split = 'train'
    df = pd.read_csv(f'../data/report_demo_{split}.csv')
    df['report_text'] = df['report_text'].str.lower().str.replace('[^\w\s]', '')
    df['report_text'] = df['report_text'].apply(extract_impression)
    df = df.loc[df['report_text']!=""]
    df = df.dropna(subset=['race'])
    # df = df[:500000]
    print(f"data shape is {df.shape}")

    # Vectorization
    print("Vectorizing")
    # vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    # X = vectorizer.fit_transform(df['report_text'])
    # X_train, X_test, y_train, y_test = train_test_split(X, df['race'], test_size=0.2, random_state=42)
    # Count frequencies
    count_vectorizer = CountVectorizer(ngram_range=(1, 1),stop_words='english')
    term_freq_matrix = count_vectorizer.fit_transform(df['report_text'])
    term_frequencies = term_freq_matrix.sum(axis=0).A1
    feature_names = count_vectorizer.get_feature_names_out()
    
    # Filter features by frequency
    # features_filtered = [feature for feature, count in zip(feature_names, term_frequencies) if count > 100]
    features_filtered = [feature for feature, count in zip(feature_names, term_frequencies) ]

    # Vectorization with filtered vocabulary
    print("Vectorizing")
    vectorizer = TfidfVectorizer(vocabulary=features_filtered, ngram_range=(1, 1), stop_words='english')
    X = vectorizer.fit_transform(df['report_text'])
    X_train, X_test, y_train, y_test = train_test_split(X, df['race'], test_size=0.2, random_state=42)

    # Logistic Regression
    print("Fitting regression")
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    # Predicting and evaluating the model
    print("Predicting")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Analyzing coefficients
    results = []
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_
    for i, class_label in enumerate(model.classes_):
        top_features = sorted(zip(coefficients[i], feature_names), reverse=True)[:20]
        print(f"Top predictive features for {class_label}:")
        for coef, feature in top_features:
            print(f"{feature}: {coef}")
            results.append({"race": class_label, "feature": feature, "coef": coef})
        
        print()

    results_df = pd.DataFrame(results)
    results_df.to_csv("../data/ln_coef_uni_impr.csv", index=False)
if __name__=='__main__':
    main()