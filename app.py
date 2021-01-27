import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.title("Predicting Alzheimers Web App")
st.sidebar.title("Model Selection Panel")

def load_data():
    data = pd.read_csv("C://Users//gursk//Desktop//Alzheimer.csv")
    return data

df=load_data()
df.drop(['Visit', 'Hand', 'MR Delay', 'Subject ID', 'MRI ID', 'CDR'], axis=1, inplace=True)

df['Group'].replace(to_replace='Nondemented', value=0, inplace=True)
df['Group'].replace(to_replace='Demented', value=1, inplace=True)
df['Group'].replace(to_replace='Converted', value=1, inplace=True)

df = pd.concat([df,pd.get_dummies(df.Gendre, prefix="Gendre", drop_first=True)], axis=1)
df.drop(['Gendre'], axis=1, inplace=True)

df["SES"].fillna(df["SES"].mode()[0], inplace=True)
df["MMSE"].fillna(df["MMSE"].mode()[0], inplace=True)

df = df[["Gendre_M", "Age", "EDUC", "SES", "MMSE", 'eTIV', "nWBV", 'ASF', "Group"]]

if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Alzheimer Raw Dataset")
    st.write(df)

if st.sidebar.checkbox("Affected by Alzheimers?"):
    st.text("""
        This web app uses machine learning to predict whether a 
        person has Alzheimer Disease or not.""")



def patient_information(gender,age,yoed,ses,mmse,etiv,nwbv,asf):

    prediction=classifier.predict([[gender,age,yoed,ses,mmse,etiv,nwbv,asf]])
    print(prediction)
    return prediction



st.sidebar.checkbox("Questionaire")
gender = st.text_input("Gender (Input 1 for Male and 0 for Female)","")
age = st.text_input("Age","")
yoed = st.text_input("Years of Education","")
ses = st.slider("Socioeconomic Status (SES) (On a scale 1 to 5)",1,5)
st.write("You've selected",ses)
mmse = st.text_input("Mini Mental State Examination (MMSE)","")
etiv = st.text_input("Estimated Total Intracranial Volume (eTIV)","")
nwbv = st.text_input("Normalize Whole Brain Volume (nWBV)","")
asf = st.text_input("Atlas Scaling Factor (ASF)","")
result = ""
if st.button("Predict"):
        result=pateint_information(gender,age,yoed,ses,mmse,etiv,nwbv,asf)