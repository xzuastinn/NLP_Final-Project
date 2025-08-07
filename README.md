# Sentiment Analysis of TripAdvisor Hotel Reviews  
_CS 6120 • Summer 2025 • Priyam Bhardwaj & Zachary Coates_

This project applies classic and neural NLP models to classify guest-written TripAdvisor hotel reviews into **positive / neutral / negative** sentiment classes.  
It reproduces and extends the experiments described in our course proposal.

---

## 1  Directory layout
Root notebook (the one you submit)
This will link to all components.
├── FinalProject_Submission.ipynb
├── data/
│   ├── tripadvisor_hotel_reviews.csv
├── preprocessing.py
├── models/
│   ├── naive_bayes.py
│   ├── logistic_regression.py
│   ├── svm.py
│   ├── mlp.py
│   └── lstm.py
├── evaluation.py
└── utils.py


## 3  Model Summary 

| Model                          | Features            | Notes                          |
| ------------------------------ | ------------------- | ------------------------------ |
| Multinomial Naive Bayes        | Unigram TF-IDF      | Fast baseline                  |
| Logistic Regression (balanced) | Uni + Bigram TF-IDF | Best macro-F1 & neutral recall |
| Linear SVM (balanced)          | Unigram TF-IDF      | Similar to LR, robust margins  |
| MLP (100 hidden, ReLU)         | Unigram TF-IDF      | Non-linear decision boundary   |
| Bi-LSTM (optional extra)       | Token indices       | Captures sequence context      |


## 2  Quick-start

### 1. Create / activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate          # or .\venv\Scripts\activate on Windows

### 2. Install Python dependencies
pip install -r requirements.txt   # see below for package list

### 3. Launch the notebook
jupyter lab         # or jupyter notebook
# open FinalProject_Submission.ipynb and run all cells ```


