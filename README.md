# AI-Based Student Performance Prediction and Analysis System

## Overview
A beginner-friendly AI/ML project that predicts student performance, estimates final marks, generates rule-based recommendations, and performs simple sentiment analysis on student feedback.

## Features
- Performance classification: Good / Average / At Risk
- Final marks prediction using Linear Regression
- Estimated pass probability
- Rule-based recommendations
- Simple NLP sentiment analyzer
- Streamlit web interface
- Dataset and model-training scripts

## Syllabus Mapping
- **CO1:** AI, intelligent-agent idea, rational decision support
- **CO2:** knowledge representation through IF-THEN rules
- **CO3:** probability/statistical data representation and feature vectors
- **CO4:** supervised learning, classification, regression, validation, overfitting discussion, bias/variance
- **CO5:** NLP and sentiment analysis; transfer learning and reinforcement learning are listed as future extensions

## Project Structure
```text
AI-Student-Performance-Predictor/
├── app.py
├── README.md
├── requirements.txt
├── dataset/student_performance.csv
├── models/
├── src/
│   ├── preprocessing.py
│   ├── classification.py
│   ├── regression.py
│   ├── rules.py
│   ├── sentiment.py
│   └── train_models.py
├── notebooks/model_training.ipynb
├── screenshots/
└── report/
```

## Installation
```bash
git clone https://github.com/your-username/AI-Student-Performance-Predictor.git
cd AI-Student-Performance-Predictor
pip install -r requirements.txt
```

## Train the Models
From the project root:
```bash
python src/train_models.py
```
This creates:
- `models/classification_model.pkl`
- `models/regression_model.pkl`

## Run the Application
```bash
streamlit run app.py
```

## Sample Input
- Study hours: 5
- Attendance: 82%
- Previous marks: 76
- Assignment marks: 18
- Internal marks: 42
- Previous failures: 0

The exact prediction depends on the trained model and dataset.

## Important Note
The included dataset is a synthetic educational dataset created for demonstration and academic project purposes. Predictions should not be treated as official academic decisions.

## Future Enhancements
- Real institutional dataset
- Cross-validation and hyperparameter tuning
- Transfer learning using a pre-trained language model
- Advanced sentiment analysis
- Teacher dashboard
- Database integration
- Reinforcement-learning-based study recommendations

## Author
**Badineni Uday Kiran**  
Integrated MTech AI
