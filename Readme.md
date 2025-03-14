Chronic Kidney Disease (CKD) Analysis and Prediction

Project Overview
This project focuses on analyzing Chronic Kidney Disease (CKD) data to gain insights into risk factors, visualize trends, and build predictive models. The dataset contains patient lab results and medical features that help classify whether an individual has CKD or not.
A Streamlit dashboard was also developed to allow users to interact with the data and explore insights visually.

Dataset Description
The dataset consists of 24 features, including numerical and categorical attributes:
Numerical Features
•	Age – Age in years
•	Blood Pressure (bp) – mm/Hg
•	Blood Glucose Random (bgr) – mg/dl
•	Blood Urea (bu) – mg/dl
•	Serum Creatinine (sc) – mg/dl
•	Sodium (sod) – mEq/L
•	Potassium (pot) – mEq/L
•	Hemoglobin (hemo) – gms
•	Packed Cell Volume (pcv)
•	White Blood Cell Count (wbcc) – cells/cumm
•	Red Blood Cell Count (rbcc) – millions/cmm
Categorical Features
•	Red Blood Cells (rbc) – (normal, abnormal)
•	Pus Cell (pc) – (normal, abnormal)
•	Pus Cell Clumps (pcc) – (present, not present)
•	Bacteria (ba) – (present, not present)
•	Hypertension (htn) – (yes, no)
•	Diabetes Mellitus (dm) – (yes, no)
•	Coronary Artery Disease (cad) – (yes, no)
•	Appetite (appet) – (good, poor)
•	Pedal Edema (pe) – (yes, no)
•	Anemia (ane) – (yes, no)
The target variable (class) indicates whether a patient has CKD or not CKD.

Data Preprocessing
1.	Handling Missing Values – Missing numerical values were filled with the median, while categorical missing values were filled with the mode.
2.	Data Cleaning – Standardized categorical labels and removed irrelevant columns.
3.	Feature Encoding – Converted categorical values to numerical using Label Encoding.
4.	Scaling – Used StandardScaler to normalize numerical features.
5.	Handling Imbalance – Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

Exploratory Data Analysis (EDA)
•	Distribution Plots – Visualized the distribution of numerical and categorical features.
•	Chi-Square Test – Analyzed relationships between categorical features and CKD status.
•	Correlation Heatmap – Examined relationships between numerical features.

Model Building
Several machine learning models were trained to predict CKD:
•	Logistic Regression
•	Random Forest Classifier
•	Support Vector Classifier (SVC)
•	XGBoost Classifier
•	Decision Tree Classifier
Model Evaluation
•	Accuracy, Precision, Recall, F1-score
•	Confusion Matrix for error analysis
•	ROC-AUC Curve for model comparison
Hyperparameter Tuning
•	Used GridSearchCV to optimize the best-performing model (Random Forest).

Streamlit Dashboard
A Streamlit app was developed for interactive visualization and exploration.
How to Run the App
1.	Clone the repository: 
git clone https://github.com/your-repo-name.git
cd your-repo-name
2.	Install dependencies: 
pip install -r requirements.txt
3.	Run the Streamlit app: 
streamlit run app.py

Technologies Used
•	Python (Pandas, NumPy, Scikit-learn, XGBoost)
•	Matplotlib & Seaborn (for visualizations)
•	Streamlit (for dashboard deployment)
•	Scipy (Chi-Square test for categorical features)
•	SMOTE (to handle data imbalance)

License
This project is open-source under the MIT License.

