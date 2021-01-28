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


def load_data():
    data = pd.read_csv("C://Users//gursk//Desktop//Alzheimer.csv")
    return data

st.title("Predicting Alzheimers Web App")
st.sidebar.title("Alzheimers information center")

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

X = df.iloc[:, :-1].values
# target 
y = df.iloc[:, -1].values

# train our data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) #20% training set

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Alzheimer Raw Dataset")
    st.write(df)

if st.sidebar.checkbox("Affected by Alzheimers?"):
    st.text("""
        Alzheimers is a type of dementia that causes a decline in memory/thinking 
        and problems with behavior. Symptoms for this type of dementia 
        generally occur slowly overtime until they become severe enough 
        to interfere with daily taks.

        Dementia is an overall term that desribes 
        a group of symptoms associated with a decline in 
        memory or other traits that may reduce the persons 
        ability to perform everyday tasks/activities. 
        Some dementia types are reversable and others have an 
        onset after a stroke or over time.
        """) 

def patient_information(gender,age,yoed,ses,mmse,etiv,nwbv,asf):

    prediction=classifier.predict([[gender,age,yoed,ses,mmse,etiv,nwbv,asf]])
    print(prediction)
    return prediction

if st.sidebar.checkbox("Questionaire"):
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
        result=patient_information(gender,age,yoed,ses,mmse,etiv,nwbv,asf)
    if result == 0:
        st.success('Alzheimers Status : {}'.format("nondemented"))
    else:
        st.success('Alzheimers Status : {}'.format('demented'))

if st.sidebar.checkbox("What is SES?"):
    st.text("""
        Socioeconomic status. This was accessed with the Hollingshead 
        four factor index ( Marital status, retired/employed status, 
        educational attainment, and occuupational prestige) and put 
        in categories ranging from 1 (highest status) to 5 (lowest status).
        """)
if st.sidebar.checkbox("What is MMSE?"):
    st.text("""
        The MMSE is used by doctors and clinicians to help access the 
        severity and progression of dementia. It is a exam consisting of 
        tests and questions and each one correctly answered adds to a  
        running tally. These questions can test anywhere from the 
        patients memory to mental abilities. Other factors are considered 
        alongside this examination and this test is also used to assess 
        changes in patients already diagnosed with dementia. Test scores
        can range from 0-30.
        """)
if st.sidebar.checkbox("What is eTIV?"):
    st.text("""
        The ICV measure, sometimes referred to as total intracranial volume 
        (TIV), refers to the estimated volume of the cranial cavity as 
        outlined by the supratentorial dura matter or cerebral contour when 
        dura is not clearly detectable. ICV is often used in studies involved 
        ith analysis of the cerebral structure under different imaging modalities, 
        such as Magnetic Resonance (MR) , MR and Diffusion Tensor Imaging (DTI), MR and 
        Single-photon Emission Computed Tomography (SPECT), Ultrasound and Computed Tomography (CT). 
        ICV consistency during aging makes it a reliable tool for correction of head size variation across 
        subjects in studies that rely on morphological features of the brain. ICV, along with age and gender are 
        reported as covariates to adjust for regression analyses in investigating progressive neurodegenerative brain 
        disorders, such as Alzheimer's disease, aging and cognitive impairment. ICV has also been utilized as an independent 
        voxel based morphometric feature to evaluate age-related changes in the structure of premorbid brain, determine characterizing 
        atrophy patterns in subjects with mild cognitive impairment (MCI) and Alzheimer's disease (AD), delineate structural abnormalities 
        in the white matter (WM) in schizophrenia, epilepsy, and gauge cognitive efficacy
        """)
if st.sidebar.checkbox("What is nWBV?"):
    st.text("""
        Normalized whole-brain volume Normalized whole-brain volume, 
        expressed as a percent of all voxels in the atlas-masked image that are labeled as gray or white matter by the automated tissue 
        segmentation process
        """)
if st.sidebar.checkbox("What is ASF?"):
    st.text("""
        Atlas scaling factor (unitless).Atlas scaling factor (unitless). Computed scaling factor that transforms native-space brain and skull 
        to the atlas target (i.e., the determinant of the transform matrix)
        """)