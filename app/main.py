import streamlit as st
import pandas as pd
import pickle as pickle
import plotly.graph_objects as go
import numpy as np
import streamlit.components.v1 as components

def get_clean_data():
    # Safely build the path relative to the current file
    data = pd.read_csv("data/breast_cancer.csv")
    
    #axis=1 means column 
    data=data.drop(["Unnamed: 32","id"],axis=1)
    
    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    
    return data

 

def add_sidebar():
    st.sidebar.header("Cell Nuclie Measurements")
    
    data=get_clean_data()
    
    sidebar_lables=[
  { "label": "Radius Mean", "column": "radius_mean" },
  { "label": "Texture Mean", "column": "texture_mean" },
  { "label": "Perimeter Mean", "column": "perimeter_mean" },
  { "label": "Area Mean", "column": "area_mean" },
  { "label": "Smoothness Mean", "column": "smoothness_mean" },
  { "label": "Compactness Mean", "column": "compactness_mean" },
  { "label": "Concavity Mean", "column": "concavity_mean" },
  { "label": "Concave Points Mean", "column": "concave points_mean" },
  { "label": "Symmetry Mean", "column": "symmetry_mean" },
  { "label": "Fractal Dimension Mean", "column": "fractal_dimension_mean" },

  { "label": "Radius SE", "column": "radius_se" },
  { "label": "Texture SE", "column": "texture_se" },
  { "label": "Perimeter SE", "column": "perimeter_se" },
  { "label": "Area SE", "column": "area_se" },
  { "label": "Smoothness SE", "column": "smoothness_se" },
  { "label": "Compactness SE", "column": "compactness_se" },
  { "label": "Concavity SE", "column": "concavity_se" },
  { "label": "Concave Points SE", "column": "concave points_se" },
  { "label": "Symmetry SE", "column": "symmetry_se" },
  { "label": "Fractal Dimension SE", "column": "fractal_dimension_se" },

  { "label": "Radius Worst", "column": "radius_worst" },
  { "label": "Texture Worst", "column": "texture_worst" },
  { "label": "Perimeter Worst", "column": "perimeter_worst" },
  { "label": "Area Worst", "column": "area_worst" },
  { "label": "Smoothness Worst", "column": "smoothness_worst" },
  { "label": "Compactness Worst", "column": "compactness_worst" },
  { "label": "Concavity Worst", "column": "concavity_worst" },
  { "label": "Concave Points Worst", "column": "concave points_worst" },
  { "label": "Symmetry Worst", "column": "symmetry_worst" },
  { "label": "Fractal Dimension Worst", "column": "fractal_dimension_worst" }
]

    input_dict={}
    
    for item in sidebar_lables:
        input_dict[item["column"]]=st.sidebar.slider(
            item["label"],
            min_value=float(0),
            max_value=float(data[item["column"]].max()),
            value=float(data[item["column"]].mean()),
        )
    return input_dict
   

def get_scaled_value(input_dict):
    data=get_clean_data()
    
    X=data.drop(['diagnosis'],axis=1)
    
    scaled_dict={}
    
    for key,value in input_dict.items():
        min_val=X[key].min()
        max_val=X[key].max()
        scaled_val=(value-min_val)/(max_val-min_val)
        #this will always return value between o and 1 
        scaled_dict[key]=scaled_val
    return scaled_dict



def get_radar_chart(input_data):
    input_data=get_scaled_value(input_data)
    
    categories = ["Radius","Texture","Perimeter","Area",
                "Smoothness","Compactness",
                "Concavity","Concave Points",
                "Symmetry","Fractal Dimension"]


    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],
            input_data['texture_mean'],
            input_data['perimeter_mean'],
            input_data['area_mean'],
            input_data['smoothness_mean'],
            input_data['compactness_mean'],
            input_data['concavity_mean'],
            input_data['concave points_mean'],
            input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']

        ],
        theta=categories,
        fill='toself',
        name='Mean value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],
            input_data['texture_se'],
            input_data['perimeter_se'],
            input_data['area_se'],
            input_data['smoothness_se'],
            input_data['compactness_se'],
            input_data['concavity_se'],
            input_data['concave points_se'],
            input_data['symmetry_se'],
            input_data['fractal_dimension_se']

        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],
            input_data['texture_worst'],
            input_data['perimeter_worst'],
            input_data['area_worst'],
            input_data['smoothness_worst'],
            input_data['compactness_worst'],
            input_data['concavity_worst'],
            input_data['concave points_worst'],
            input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']

        ],
        theta=categories,
        fill='toself',
        name='Worst value'
    ))
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )
    return fig

def add_predictions(input_data):
    model=pickle.load(open("model/model.pkl","rb"))
    scaler=pickle.load(open("model/scaler.pkl","rb"))
    
    input_array= np.array(list(input_data.values())).reshape(1,-1)
    scaled_array=scaler.transform(input_array)
    prediction=model.predict(scaled_array)
    
    st.subheader("Cell Cluster Prediction")
    st.write("The model predicts that the breast mass is: ")
    
    if prediction[0]==0:
        st.write("<span class='diagnosis benign'>Benign</span>",unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>Malignant</span>",unsafe_allow_html=True)
    
    st.write("Probability of being Benign: ", model.predict_proba(scaled_array)[0][0])
    st.write("Probability of being Malignant: ", model.predict_proba(scaled_array)[0][1])
    
    st.write("This app is not a replacement for professional medical advice. Please consult a healthcare professional for accurate diagnosis and treatment options.")

    
def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction",
        page_icon=":female doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    

    
input_data=add_sidebar()
    
with st.container():
    st.title("# Breast Cancer Prediction App")
    st.write("Please connect this app to the cytology lab to help diagnose breast cancer .Please make sure the predictions made by the app is completely the results of a trained model using machine learning the app predicts that whether the breast mass is malignant or benign based on the measurements it receives from the cytology lab you can also enter the measurements manually using the sliders given in side bar.")
        
    col1,col2= st.columns([4,1],gap="medium",border=True)
    with col1:
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        
    with col2:
        add_predictions(input_data)
   
    # st.html("pages/index.html")
    # st.write("# Breast Cancer Prediction App")
    



if __name__=='__main__':
    main()