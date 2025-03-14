import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import pickle
import seaborn as sns
import streamlit.components.v1 as components


with open('best_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Chronic Kidney Prediction')

st.image(f"kidney.jpg")

kidney_to_wiki = 'https://en.wikipedia.org/wiki/Kidney'

st.markdown(f'### Learn more about kidney at'
            f"[ wikipedia]({kidney_to_wiki})!")

st.title('Distributon of the kidney Datasets')

kidney = pd.read_csv('chronic_kidney_disease.csv')

st.header('The Datasets')
kidney

kidney_clean =pd.read_csv('clean_chronic_kidney_disease.csv')

st.header('The datasets after cleaning and preprocessing')
kidney_clean

st.header('how many people has chronic kidney disease')
kidney_diease_value_count = kidney_clean['class'].value_counts()

fig, ax = plt.subplots()
kidney_diease_value_count.plot.bar(ax=ax, color=['blue', 'red'])
st.pyplot(fig)

st.markdown(f"**class 0** = No Chronic Kidney disease, **class 1** = has Chronic Kidney disease")

st.sidebar.subheader("📊 Select Feature to Compare")

numerical_cols = numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 
                'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

selected_feature = st.sidebar.selectbox("Select a Numerical Feature:", numerical_cols)
plot_type = st.sidebar.radio("Choose Plot Type:", ["Boxplot", "KDE Plot"])

st.subheader(f"Distribution of {selected_feature} by Class")

fig, ax = plt.subplots(figsize=(8, 5))

if plot_type == "Boxplot":
    sns.boxplot(x=kidney_clean["class"], y=kidney_clean[selected_feature], ax=ax, palette="coolwarm")
    ax.set_title(f"Boxplot of {selected_feature} by Class")
else:
    sns.kdeplot(data=kidney_clean, x=selected_feature, hue="class", fill=True, common_norm=False, palette="coolwarm")
    ax.set_title(f"KDE Distribution of {selected_feature}")

ax.set_xlabel(selected_feature)
ax.set_ylabel("Density" if plot_type == "KDE Plot" else selected_feature)
st.pyplot(fig)

st.header('categorical features with most effect on the class value')

categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 
                    'appet', 'pe', 'ane', 'class']

def chi_square_with_class(data, categorical_cols, target_col='class'):
    results = []

    for col in categorical_cols:
        if col != target_col: 
            table = pd.crosstab(data[col], data[target_col])
            chi2, p, dof, expected = chi2_contingency(table)
            results.append((col, target_col, chi2, p))

    results_data = pd.DataFrame(results, columns=['Variable', 'Compared with', 'Chi-Square', 'p-value'])
    
    return results_data


chi_results = chi_square_with_class(kidney_clean, categorical_cols)
chi_results = chi_results.sort_values(by='p-value')
st.subheader("Chi-Square Test Results")
st.dataframe(chi_results)

st.subheader("Chi-Square Values for Categorical Features vs Class")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=chi_results['Variable'], y=chi_results['Chi-Square'], palette='viridis', ax=ax)
plt.xticks(rotation=45)
plt.xlabel("Categorical Variables")
plt.ylabel("Chi-Square Score")
plt.title("Chi-Square Values for Categorical Features vs Class")
st.pyplot(fig)

components.html("", height=750)
