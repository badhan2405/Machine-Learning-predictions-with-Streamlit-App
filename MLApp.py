import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
#matplotlib.use("Agg")

from PIL import Image


# Set title

st.title('Machine learning App')
image = Image.open('l.jpg')
st.image(image,use_column_width=True)

def main():
    activities=["EDA","Visualisation","Model","About us"]
    option=st.sidebar.selectbox("Select option:",activities)

# Dealing with EDA part
    if option=="EDA":
        st.subheader("Exploratory Data Analysis")

        data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox("Display Shape"):
                st.write(df.shape)
            if st.checkbox("Display Columns"):
                st.write(df.columns)
            if st.checkbox("Select multiple columns"):
                selected_columns=st.multiselect("Select preferred columns:",df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            
            if st.checkbox("Display summary"):
                st.write(df.describe().T)
            if st.checkbox("Display Null Values"):
                st.write(df.isnull().sum())
            if st.checkbox("Display the data types"):
                st.write(df.dtypes)
            if st.checkbox("Display Correlation of data various columns"):
                st.write(df.corr())        


             



# DEALING WITH THE VISUALISATION PART  
  
    elif option=="Visualisation":
        st.subheader("Visualisation")

        data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(50))

        if st.checkbox("Select Multiple columns to plot"):
            selected_columns=st.multiselect("Select your preferred columns,",df.columns)
            df1=df[selected_columns]
            st.dataframe(df1) 

        if st.checkbox("Display HeatMap"):
            st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
            st.pyplot()
        if st.checkbox("Display pairplot"):
            st.write(sns.pairplot(df1,diag_kind="kde",))
            st.pyplot()
        if st.checkbox("Display Pie Chart"):
            all_columns=df.columns.to_list()
            pie_columns=st.selectbox("Select the column to display",all_columns)
            pieChart=df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pieChart)
            st.pyplot()              


    
# DEALING WITH THE MODEL BUILDING PART

    elif option=="Model":
        st.subheader("Model Building")

        data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(50))

        if st.checkbox("Select Multiple columns"):
            new_data=st.multiselect("Select your preferred columns. NB: Let your target variable be the last column to be selected ",df.columns)
            df1=df[new_data]
            st.dataframe(df1)

        # Dividing my data into X and y variables
        
            X=df1.iloc[:,0:-1]
            y=df1.iloc[:,-1]
         
        seed=st.sidebar.slider("Seed",1,200)
        
        classifier_name=st.sidebar.selectbox("Select your preferred classifier:",('KNN','SVM','Logistic Regression',"Naive Bayes","Decision Tree"))
        

        def add_parameter(name_of_clf):
            params=dict()
            if name_of_clf=="SVM":
                C=st.sidebar.slider("C",0.01,15.0)
                params["C"]=C 
            else:
                name_of_clf="KNN"
                K=st.sidebar.slider("K",1,15)
                params["K"]=K
            return params

        params=add_parameter(classifier_name)


        #Accessing our Classifier

        def get_classifier(name_of_clf,params):
            clf=None
            if name_of_clf=="SVM":
                clf=SVC(C=params['C'])
            elif name_of_clf=='KNN':
                clf=KNeighborsClassifier(n_neighbors=params["K"])
            elif name_of_clf=="Logistic Regression":
                clf=LogisticRegression()
            elif name_of_clf=="Naive Bayes":
                clf=GaussianNB()
            elif name_of_clf=="Decision Tree":
                clf=DecisionTreeClassifier()
            else:
                st.warning("you didn't select any option, please select at least one algorithm")
            return clf        
    

        clf=get_classifier(classifier_name,params)
    
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=seed)

        clf.fit(X_train,y_train)

        y_pred=clf.predict(X_test)
        st.write('Predictions:',y_pred)
    
        accuracy=accuracy_score(y_test,y_pred)

        st.write('Name of classifier:',classifier_name)
        st.write('Accuracy',accuracy)

# Dealing with the about us page

    elif option=="About us":
        st.markdown('This is an interactive web page for our ML project, feel free to use it. This dataset is fetched from the UCI Machine learning repository. The analysis in here is to demonstrate how we can present our wok to our stakeholders in an interractive way by building a web app for our machine learning algorithms using different dataset.'
            )


        st.balloons()
    #   ..............
    

if __name__ == '__main__':
    main() 