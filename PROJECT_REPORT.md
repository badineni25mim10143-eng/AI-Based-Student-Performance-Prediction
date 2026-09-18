# AI-BASED STUDENT PERFORMANCE PREDICTION AND ANALYSIS SYSTEM

## Project Report

**Student:** Badineni Uday Kiran  
**Program:** Integrated MTech AI  
**Academic Year:** 2026  
**Project Type:** Artificial Intelligence and Machine Learning

---

## ABSTRACT

The AI-Based Student Performance Prediction and Analysis System is an educational artificial intelligence project developed to demonstrate how machine learning, rule-based reasoning, probability, data representation, and natural language processing can be combined in a single application.

The system accepts academic features such as study hours, attendance, previous marks, assignment marks, internal marks, and previous failures. A supervised classification model predicts a performance category of Good, Average, or At Risk. A linear regression model estimates final marks. The system also provides rule-based recommendations and a simple sentiment analyzer for student feedback.

The project is implemented in Python using Pandas, NumPy, Scikit-learn, Joblib, and Streamlit. A synthetic dataset is included so the project can be executed without collecting personal student information. The project demonstrates the major concepts in the provided AI syllabus while remaining simple enough for academic demonstration.

---

# CHAPTER 1: INTRODUCTION

## 1.1 Introduction to Artificial Intelligence

Artificial Intelligence is a field of computer science concerned with developing systems that can perform tasks associated with intelligent behavior. These tasks include learning from data, reasoning, problem solving, perception, language processing, and decision support.

In this project, AI is applied to educational data. The system receives information about a student, processes the information using machine learning models and rules, and produces predictions and recommendations.

## 1.2 Problem Statement

Students and teachers often have multiple academic factors to consider when understanding performance. Attendance, study time, previous marks, assignments, internal assessment, and previous failures can all provide useful information.

The problem addressed by this project is to develop a small AI system that can use these features to estimate student performance and provide understandable recommendations.

## 1.3 Objectives

1. Develop an AI-based student performance application.
2. Build a supervised classification model.
3. Predict approximate final marks using regression.
4. Estimate a model-based probability associated with the predicted class.
5. Implement simple knowledge-based rules.
6. Analyze student feedback using sentiment analysis.
7. Present results through a simple web interface.
8. Demonstrate concepts from CO1 to CO5.

## 1.4 Scope

The project is intended for educational demonstration. It is not designed to replace teachers, examinations, or institutional evaluation. The current version uses a synthetic dataset.

---

# CHAPTER 2: AI AND INTELLIGENT AGENTS

## 2.1 AI Background

AI systems generally combine data, algorithms, representations, and decision procedures. Modern AI includes machine learning, deep learning, natural language processing, computer vision, and reinforcement learning.

## 2.2 Intelligent Agent

An intelligent agent perceives information from its environment and performs actions based on that information.

In this project:

- **Environment:** Student academic information and feedback.
- **Percepts:** Attendance, marks, study hours, failures, and feedback.
- **Processing:** ML prediction, regression, rules, and sentiment analysis.
- **Action:** Display prediction and recommendations.

## 2.3 Rationality

A rational decision-support agent selects an action based on the available information and its objective. The project's recommendation component uses available student information to produce practical study suggestions.

---

# CHAPTER 3: PROBLEM SOLVING AND KNOWLEDGE REPRESENTATION

## 3.1 Problem Solving

The project treats performance prediction as a data-driven problem:

Input → Preprocessing → Model → Prediction → Recommendation.

## 3.2 Knowledge Representation

Simple knowledge is represented with IF-THEN rules.

Example:

IF attendance < 75  
THEN recommend improving attendance.

IF study hours < 3  
THEN recommend increasing regular study time.

IF previous failures > 0  
THEN recommend extra attention to previously failed subjects.

## 3.3 Propositional Logic Concept

Rules can be viewed as logical conditions. For example:

AttendanceBelow75 AND StudyHoursBelow3 → AtRiskRecommendation.

This demonstrates the basic idea of representing knowledge as conditions and conclusions.

## 3.4 Prolog Extension

A future version could represent these rules in Prolog, for example:

```prolog
at_risk(Attendance, StudyHours) :-
    Attendance < 75,
    StudyHours < 3.
```

The current implementation uses Python rules so that the project remains easy to run.

---

# CHAPTER 4: PROBABILITY, STATISTICS AND DATA REPRESENTATION

## 4.1 Probability

The classification model provides class probabilities through `predict_proba`. These values represent the model's estimated probabilities for the available classes.

## 4.2 Random Variables

Student attributes can be represented as random variables. Examples include study hours, attendance, previous marks, and final marks.

## 4.3 Mean and Variance

Descriptive statistics can be used to understand the distribution of numerical features. Mean provides the average value, while variance describes how widely values are spread around the mean.

## 4.4 Data Representation

Each student is represented as a feature vector:

[study_hours, attendance, previous_marks, assignment_marks, internal_marks, previous_failures]

This vector is provided to the machine learning model.

---

# CHAPTER 5: MACHINE LEARNING BASICS

## 5.1 Machine Learning

Machine learning enables a computer program to learn patterns from examples rather than relying entirely on manually written rules.

## 5.2 Supervised Learning

This project uses supervised learning because the training data contains target values:

- Performance category for classification.
- Final marks for regression.

## 5.3 Classification

A Decision Tree classifier is used to predict:

- Good
- Average
- At Risk

Decision trees repeatedly divide the feature space using conditions that help separate classes.

## 5.4 Regression

Linear Regression is used to estimate final marks from student features. The model learns a relationship between input variables and a continuous target.

## 5.5 Training and Testing

The dataset is split into training and testing subsets. The training data is used to fit the model, while the testing data is used to evaluate generalization to unseen records.

## 5.6 Overfitting and Underfitting

Overfitting occurs when a model learns training-specific patterns too closely and performs poorly on unseen data. Underfitting occurs when a model is too simple to capture important patterns.

The Decision Tree is limited with a maximum depth in this project to reduce unnecessary complexity.

## 5.7 Bias and Variance

Bias represents error caused by overly simple assumptions. Variance represents sensitivity to the particular training data. Model selection attempts to achieve a useful balance.

## 5.8 Hyperparameters

The Decision Tree uses `max_depth=4` as a hyperparameter. A future version can use cross-validation and grid search to tune hyperparameters systematically.

---

# CHAPTER 6: SYSTEM DESIGN

## 6.1 Architecture

```text
+----------------------+
| Student Information  |
+----------+-----------+
           |
           v
+----------------------+
| Data Representation  |
+----------+-----------+
           |
           v
+----------------------+
| ML Classification    |
+----------+-----------+
           |
           +------------------+
           |                  |
           v                  v
+----------------+   +------------------+
| Performance    |   | Regression Model |
| Classification |   | Final Marks      |
+--------+-------+   +--------+---------+
         |                    |
         +----------+---------+
                    v
          +--------------------+
          | Recommendation     |
          | Rule Engine        |
          +---------+----------+
                    |
                    v
          +--------------------+
          | Streamlit Dashboard|
          +--------------------+

Student Feedback
       |
       v
+--------------------+
| Sentiment Analyzer |
+--------------------+
       |
       v
Positive / Negative / Neutral
```

## 6.2 Functional Requirements

- Accept student academic inputs.
- Predict performance.
- Estimate final marks.
- Display probability.
- Generate recommendations.
- Analyze feedback sentiment.
- Display dataset information.

## 6.3 Non-Functional Requirements

- Easy to use.
- Simple interface.
- Fast prediction.
- Modular Python code.
- Reproducible training.

---

# CHAPTER 7: IMPLEMENTATION

## 7.1 Technologies

Python is the main programming language.

Libraries:
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## 7.2 Dataset

The included dataset contains 180 synthetic student records. The data was generated for academic demonstration and does not represent real students.

Features:
- study_hours
- attendance
- previous_marks
- assignment_marks
- internal_marks
- previous_failures
- final_marks
- performance

## 7.3 Preprocessing

The project loads the CSV using Pandas. The selected feature columns are separated from the target variable before model training.

## 7.4 Classification Implementation

A Decision Tree Classifier is trained using the academic features. The trained model is saved using Joblib.

## 7.5 Regression Implementation

A Linear Regression model is trained to estimate final marks. Mean Absolute Error, Root Mean Squared Error, and R² are calculated during training.

## 7.6 Recommendation Engine

Python IF-THEN conditions generate understandable suggestions based on attendance, study time, marks, assignments, and failures.

## 7.7 Sentiment Analysis

The current sentiment analyzer uses a small educational word list. Positive and negative words are counted to classify feedback as Positive, Negative, or Neutral.

This is intentionally simple and is not intended to compete with modern language models.

---

# CHAPTER 8: RESULTS AND TESTING

## 8.1 Classification Testing

The classification program prints:
- Accuracy
- Classification report
- Confusion matrix

The exact metrics are produced when the included dataset is used to train the model.

## 8.2 Regression Testing

The regression program prints:
- MAE
- RMSE
- R²

These metrics provide different views of prediction error and model fit.

## 8.3 Sample Test Case

Input:

Study Hours = 5  
Attendance = 82%  
Previous Marks = 76  
Assignment Marks = 18  
Internal Marks = 42  
Previous Failures = 0

The application produces a model-generated performance category, predicted final marks, and estimated probability. Because the result is generated by the trained model, the exact numerical output should be taken from the application rather than hard-coded in the report.

## 8.4 Sentiment Test

Input:

"I am happy with my performance and understand the subjects well."

Expected educational demonstration output:

Positive

Input:

"I am struggling with programming and feel confused."

Expected educational demonstration output:

Negative

---

# CHAPTER 9: CO-WISE SYLLABUS MAPPING

## CO1: Introduction to AI

The project demonstrates AI concepts through intelligent decision support, rational recommendations, agents and environments.

## CO2: Problem Solving and Knowledge Representation

The rule engine represents knowledge using IF-THEN conditions. The report also discusses propositional logic and a possible Prolog extension.

## CO3: Classical Statistics to ML

The project uses numerical data representation, feature vectors, probability outputs, and statistical evaluation.

## CO4: Machine Learning Basics

The main implementation demonstrates supervised learning, classification, regression, model validation, hyperparameters, overfitting, underfitting, and bias-variance concepts.

## CO5: Case Studies

The sentiment analyzer demonstrates an NLP application. Transfer learning and reinforcement learning are included as future extensions rather than being unnecessarily added to the core beginner project.

---

# CHAPTER 10: LIMITATIONS

1. The dataset is synthetic.
2. The model should not be used for official academic decisions.
3. The sentiment analyzer uses a small word list.
4. The project does not currently use deep learning or transfer learning.
5. The model's performance depends on the dataset.
6. More reliable evaluation would require larger real-world datasets and cross-validation.

---

# CHAPTER 11: FUTURE ENHANCEMENTS

- Use a larger real-world or institution-approved dataset.
- Add cross-validation.
- Add hyperparameter tuning.
- Compare multiple classifiers.
- Use transformer-based sentiment analysis.
- Add transfer learning with pre-trained models.
- Build a teacher dashboard.
- Add database storage.
- Add student progress tracking.
- Explore reinforcement learning for personalized study recommendations.

---

# CONCLUSION

The AI-Based Student Performance Prediction and Analysis System demonstrates the practical use of Artificial Intelligence and Machine Learning in education.

The project combines supervised classification, regression, probability outputs, knowledge-based reasoning, data representation, and NLP sentiment analysis in a single application.

The modular structure makes the project easy to understand, execute, demonstrate, and extend. It also provides a direct connection between the implemented features and the concepts covered in CO1 to CO5 of the Artificial Intelligence syllabus.

---

# REFERENCES

1. Stuart Russell and Peter Norvig, Artificial Intelligence: A Modern Approach.
2. Scikit-learn documentation.
3. Python documentation.
4. Pandas documentation.
5. Streamlit documentation.
6. Natural Language Processing introductory resources.

---

# APPENDIX A: HOW TO RUN

```bash
pip install -r requirements.txt
python src/train_models.py
streamlit run app.py
```

# APPENDIX B: GITHUB SUBMISSION

1. Create a new GitHub repository named `AI-Student-Performance-Predictor`.
2. Upload all project files.
3. Ensure `README.md` is visible on the repository home page.
4. Add screenshots of the running application to the `screenshots` folder.
5. Add the final PDF report to the `report` folder.
6. Test the project on another computer before submission.
