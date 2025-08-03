import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle

#creating model 
def create_model(data):
    X=data.drop(['diagnosis'],axis=1)
    y=data['diagnosis']

    #Normarlize data /scale data
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    
    #split into train test 
    X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)
    
    #train
    model=LogisticRegression()
    model.fit(X_train,y_train)
    
    #test 
    y_pred=model.predict(X_test)
    print("The accuracy of the model is :" ,accuracy_score(y_test,y_pred))
    print("Classification Report for the model :\n" , classification_report(y_test,y_pred))
    return model ,scaler

def get_clean_data():
    # Safely build the path relative to the current file
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'breast_cancer.csv')
    data = pd.read_csv(data_path)
    
    #axis=1 means column 
    data=data.drop(["Unnamed: 32","id"],axis=1)
    
    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    
    return data

    

def main():
    data=get_clean_data()
    
    model,scaler=create_model(data)
    
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)
    with open('scaler.pkl','wb') as f: 
        pickle.dump(scaler,f)
    
    
if __name__=='__main__':
    main()