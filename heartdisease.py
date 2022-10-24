import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import altair as alt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# import dataset
heart_df = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/midterm_object/code/heart.csv")
hd_Df =  heart_df.copy()
conti = ["trestbps", "chol", "thalach", "oldpeak"]
for i in conti:
    q1 = np.quantile(hd_Df[i], .25)
    q3 = np.quantile(hd_Df[i], .75)
    iqr = q3 - q1
    max = q3 + 1.5 * iqr
    min = q1 - 1.5 * iqr
    hd_Df = hd_Df.where(hd_Df[i] <= max).where(hd_Df[i] >= min).dropna()

q1_old = np.quantile(hd_Df["oldpeak"], .25)
q3_old = np.quantile(hd_Df["oldpeak"], .75)
iqr_old = q3_old - q1_old
max_old = q3_old + 1.5 * iqr_old
min_old = q1_old - 1.5 * iqr_old
hd_Df = hd_Df.where(hd_Df["oldpeak"] <= max_old).where(hd_Df["oldpeak"] >= min_old).dropna()

hd_df = hd_Df.drop(columns="fbs")
Y_df = hd_df["target"].values.copy()
X_df = hd_df.iloc[:, 0:12].values.copy()
X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=1)
Y_train = Y_train.reshape(Y_train.shape[0], 1)
Y_test = Y_test.reshape(Y_test.shape[0], 1)
knnmodel_1 = KNeighborsClassifier(n_neighbors=3)
knnmodel_1.fit(X_train, Y_train)
test_predict = knnmodel_1.predict(X_test)
test_confusion = metrics.confusion_matrix(Y_test, test_predict)
tn_tr, fp_tr, fn_tr, tp_tr = test_confusion.ravel()
test_error_rate = (fn_tr + fp_tr)/(tn_tr+fp_tr + fn_tr + tp_tr)

target0 = hd_Df.where(hd_Df["target"] == 0).dropna()
target1 = hd_Df.where(hd_Df["target"] == 1).dropna()
names = list(hd_Df.columns)
names.pop(-1)
ex_df = pd.DataFrame(index=["with heart diseases", "without heart disease"])
for i in names:
    ex_df[i] = [target1[i].mean(), target0[i].mean()]

data = st.container()
preprocess = st.container()
EDA = st.container()
summary = st.container()
prediction = st.container()
reference = st.container()




with st.sidebar:
    st.title("Heart Diseases Prediction")
    add_radio = st.radio("NAVIGATOR", ("data introduction", "data analysis", "prediction"))

    if add_radio == "data introduction":
        with data:
            st.title("Introduction of Data")
            st.text("The heart disease dataset comes from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
            st.markdown(f"<div style='text-align: center;'>{'table1 the first few rows of heartdisease dataset'}</div>", unsafe_allow_html=True)
            st.write(heart_df.head())
            st.text("""- age: in years
- sex: 1-male; 0-female
- cp: chest pain type(0, 1, 2, 3)
- trestbps: resting blood pressure (in mm Hg on admission to the hospital)
- chol: serum cholestoral in mg/dl
- fbs: fasting blood sugar > 120 mg/dl (true:1; false:0)
- restech: resting electrocardiographic results (0,1,2)
- thalach: maximum heart rate achieved
- exang: exercise induced angina (1 = yes; 0 = no)
- oldpeak: ST depression induced by exercise relative to rest
- slope: the slope of the peak exercise ST segment
- ca: number of major vessels (0-3) colored by flourosopy
- thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
- target: have heart disease = 1; no = 0""")

    if add_radio == "data analysis" :
        with preprocess:
            st.header("preprocess")
            st.markdown("* visualize the missingdata")
            miss_visual, ax = plt.subplots()
            sns.heatmap(heart_df.isna(), ax=ax).set_title("image1 missing data visualization")
            st.pyplot(miss_visual)
            st.write("Image 1 shows that there is no missing data in the dataset.")

            st.markdown("* check the outlier")
            st.write("There are 4 attributes with continuous values. Image 2 shows all of them have outliers.")
            outlier, (ax1, ax2, ax3, ax4)=plt.subplots(1, 4, figsize=(32, 8))
            sns.boxplot(y=heart_df["trestbps"], ax=ax1).set_title("trestbps")
            sns.boxplot(y=heart_df["chol"], ax=ax2).set_title("chol")
            sns.boxplot(y=heart_df["thalach"], ax= ax3).set_title("thalach")
            sns.boxplot(y=heart_df["oldpeak"], ax=ax4).set_title("oldpeak")
            outlier.suptitle("image2 checking the outlier", size=32)
            st.pyplot(outlier)
            st.write("According to the following equations,  the outlier is deleted.")
            st.write("$Q_3$ is the number at 0.75 quantile.")
            st.write("$Q_1$ is the number at 0.25 quantile.")
            st.latex(r'''IQR = Q_3 - Q_1''')
            st.latex(r'''max = Q_3 + 1.5 \times IQR ''')
            st.latex(r'''min = Q_1 - 1.5 \times IQR''')
            outlier_after, (axi1, axi2, axi3, axi4)=plt.subplots(1, 4, figsize=(32, 8))
            axi1 = sns.boxplot(y=hd_Df["trestbps"], ax=axi1).set_title("trestbps")
            axi2 = sns.boxplot(y=hd_Df["chol"], ax=axi2).set_title("chol")
            axi3 = sns.boxplot(y=hd_Df["thalach"], ax= axi3).set_title("thalach")
            axi4 = sns.boxplot(y=hd_Df["oldpeak"], ax=axi4).set_title("oldpeak")
            outlier_after.suptitle("image3 checking after deleting the outliers",size=32)
            st.pyplot(outlier_after)
            st.write("Image 3 shows there is no outlier after the above processing.")

        with EDA:
            st.header("EDA")
             # expected value table
            st.markdown("* the expected value of different attributes")
            st.markdown(f"<div style='text-align: center;'>{'table2 the expected value'}</div>", unsafe_allow_html=True)
            st.write(ex_df.style.set_properties(**{'background-color': 'yellow'}, subset=["cp", "thalach", "exang", "oldpeak", "ca"]))
            st.write("The attibutes highlighted by yellow have great differences between the expected value of people with heart disease and without heart disease.\
                The table shows people with heart diseas have higher chest pain, higher maximum heart rate achieved, lower number of major vessels, ST depression and \
                    most of them don't have exercise induced angina.")
            # correlation
            st.markdown("* the correlation plot")
            fig = plt.figure(figsize=(16, 16))
            corr = sns.heatmap(hd_Df.corr(), annot=True).set_title("Imag4 correlation map", fontsize=30)
            st.pyplot(fig)            
            st.write("In the attributes with continuous value, old peak and thalach have high absolute \
                    correlations, age, trestbps and chol's correlations are relatively low. ")
            st.write("In the attibutes with discrete value, fbs only have two values, but the\
                     correlation with target is only -0.041, so I can confirm that fbs cannot help to predict the\
                    heart disease. Sex, cp, thal, exang, slope and ca have high absolute correlations \
                    with target, and restecg's absolute correlation is relatviely low.")

            # 2d_scatter plot about the old peak and thalch
            st.markdown("* the distribution of attributes with continuous values")
            st.write("In the attributes with continuous values, old peak and thalach have high absolute correlations, \
                so these two attributes are choses to draw the 2d plot")
            fig = alt.Chart(hd_Df, title="Image5 the scatter plot between old peak and thalach").mark_point().encode(
                x='thalach',
                y='oldpeak',
                color='target',
                tooltip = ["thalach", "oldpeak", "target"]
                ).interactive()
            st.altair_chart(fig)
            st.write("the difference between the distribution of target=1 and target=0 is not really obvious. Only desity difference exists.\
                So another variable should be added to the plot to check whether this dataset fit the classification problem. Besides ca, exang and cp alse have high absolute correlations, and high difference of the expected values between people with heart disease and without heart disease,\
                So next following two plots are about old peak, thalach and exang, old peak, thalach and cp and old peak, thalach and ca.")

            # 3d_scatter plot among old peak, thalach and exang
            st.write("Add exang to the before plot and produce a 3-d scatter plot.")
            fig = px.scatter_3d(hd_Df, x='thalach', y='oldpeak', z='exang', color='target', title="Image6 3d plot between thalach, oldpeak and exang")
            st.plotly_chart(fig)
    
            # 3d_scatter plot among old peak, thalch and cp
            st.write("Add cp to the before plot and produce a 3-d scatter plot.")
            fig = px.scatter_3d(hd_Df, x='thalach', y='oldpeak', z='cp', color='target', title="Image7 3d plot between thalch, oldpeak and cp")
            st.plotly_chart(fig)

            # 3d_scatter plot among old peak, thalch and ca
            st.write("Add ca to the before plot and produce a 3-d scatter plot.")
            fig = px.scatter_3d(hd_Df, x='thalach', y='oldpeak', z='ca', color='target', title="Image8 3d plot between thalch, oldpeak and ca")
            st.plotly_chart(fig)
            st.write("In those 3d-plots, the distribution of target=1 alse have obviously great difference from distribution of target=0.")




            # discrete value
            st.markdown("* The distributions of attributes with discrete values")
            st.write("only the variables having high correlations with target are drawn")
            add_selectbox2 = st.selectbox("choose one", ("exang", "cp", "ca", "slope", "sex"))
            if add_selectbox2 == "exang":
                target0_ex = pd.DataFrame(target0["exang"].value_counts()).rename(columns={"exang": "target0"})
                target1_ex = pd.DataFrame(target1["exang"].value_counts()).rename(columns={"exang": "target1"})
                target_ex = pd.merge(target0_ex, target1_ex, left_index=True, right_index=True)
                st.bar_chart(target_ex)
                st.write("when exang=0, the proportion of people with heart disease: 0.67; when exang=1, the proportion of people with heart disease: 0.21.")

            if add_selectbox2 == "cp":
                target0_cp = pd.DataFrame(target0["cp"].value_counts()).rename(columns={"cp": "target0"})
                target1_cp = pd.DataFrame(target1["cp"].value_counts()).rename(columns={"cp": "target1"})
                target_cp = pd.merge(target0_cp, target1_cp, left_index=True, right_index=True)
                st.bar_chart(target_cp)
                st.write("when cp=0, the proportion of people with heart disease: 0.25; when cp=1, the proportion of \
                        people with heart disease: 0.80; when cp=2, the proportion of people with heart disease: 0.77; when cp=3, \
                        the proportion of people with heart disease: 0.66")

            if add_selectbox2 == "ca":
                target0_ca = pd.DataFrame(target0["ca"].value_counts()).rename(columns={"ca": "target0"})
                target1_ca = pd.DataFrame(target1["ca"].value_counts()).rename(columns={"ca": "target1"})
                target_ca = pd.merge(target0_ca, target1_ca, left_index=True, right_index=True)
                st.bar_chart(target_ca)
                st.write("when ca=0, the proportion of people with heart disease: 0.72; \
                        when ca=1, the proportion of people with heart disease: 0.29; \
                        when ca=2, the proportion of people with heart disease: 0.16; \
                        when ca=3, the proportion of people with heart disease: 0.13; \
                        when ca=4, the proportion of people with heart disease: 0.83; \
                        Actueally, the relation between ca and target are more obvious than it was shown in correlation, because it is pit-like. ")

            if add_selectbox2 == "slope":
                target0_slo = pd.DataFrame(target0["slope"].value_counts()).rename(columns={"slope": "target0"})
                target1_slo = pd.DataFrame(target1["slope"].value_counts()).rename(columns={"slope": "target1"})
                target_slo = pd.merge(target0_slo, target1_slo, left_index=True, right_index=True)
                st.bar_chart(target_slo)
                st.write("when slope=0, the proportion of people with heart disease: 0.38; \
                        when slope=1, the proportion of people with heart disease: 0.33; \
                        when slope=2, the proportion of people with heart disease: 0.72; ")

            if add_selectbox2 == "sex":
                target0_sex = pd.DataFrame(target0["sex"].value_counts()).rename(columns={"sex": "target0"})
                target1_sex = pd.DataFrame(target1["sex"].value_counts()).rename(columns={"sex": "target1"})
                target_sex = pd.merge(target0_sex, target1_sex, left_index=True, right_index=True)
                st.bar_chart(target_sex)
                st.write("Female has higher risk of heart disease.")

        with summary:
            st.header("summary")
            st.write("According to the image 6, image 7 and all the bar charts among the variables with discrete values,\
                the distribution of target=0 have great difference from target=1, which means this dataset is good for classification.\
                    But the relations between heart disease and predictors are all really complex, it is so hard to conclude them as simple linear trends, because, heart disease is multi-factor induced.")
    
    
    
    if add_radio == "prediction":
        with prediction:
            st.header("KNN classifier model")
            st.write("As what I said in the previous chapter, heart disease is a multi-factor induced disease, So I decided the predict the existence of heart disease according to the probability that it exists.")
            st.write("KNN model is a classical model for classification problem, the test observation is classified to the class with the largest probability from the following eqaution.")
            st.latex(r'''Pr(Y=j|X=x_0) = \frac{1}{K} \sum_{i \in N_0} I(y_i = j)''')
            st.write("The yellow area in the following table is Y, and the green area is X. Because fbs has really low absolute correlation with targe, this is dropped.")
            st.markdown(f"<div style='text-align: center;'>{'table3 the division of X and Y'}</div>", unsafe_allow_html=True)
            st.write(hd_df.head().style.set_properties(**{'background-color': 'yellow'}, subset=['target'])\
                .set_properties(**{'background-color': 'green'}, subset=['sex', "age", "cp", "trestbps", "chol", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]))
            st.write("In this prediction, set K=3")
            
            st.write("confusion matrix: ")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Positive", tp_tr)
            col2.metric("True Negative", tn_tr)
            col3.metric("False Positive", fp_tr)
            col4.metric("False Negative", fn_tr)
            st.write(f"train error rate : {test_error_rate: .3f}")
            st.write("This model has high accuracy.")
            st.subheader("user defined prediction")
            age = st.slider("your age", value=50, min_value=20, max_value=100)
            choose_sex = st.radio("your sex", ("female", "male"))
            if choose_sex == "female":
                sex = 0
            if choose_sex == "male":
                sex = 1
            cp = st.radio("your chest pain type", (0, 1, 2, 3))
            trestbps = st.slider("your resting blood presure(trestbps)", value=100, min_value=90, max_value=200)
            chol = st.slider("your serum cholestoral(chol)", value=300, min_value=100, max_value=500)
            restecg = st.radio("your resting electrocardiographic results(restecg)", (0, 1, 2))
            thalach = st.slider("your maximum heart rate achieved(thalach)", value=135, min_value=70, max_value=200)
            choose_exang = st.radio("your exercise induced angina(exang)", ("yes", "no"))
            if choose_exang == "yes":
                exang = 1
            if choose_exang == "no":
                exang = 0
            oldpeak = st.slider("your ST depression induced by exercise relative to rest(oldpeak)", value=3.25, min_value=0.0, max_value=6.5)
            slope = st.radio("your slope of the peak exercise ST segment(slope)", (0, 1, 2))
            ca = st.radio("your number of major vessels colored by flourosopy(ca)", (0, 1, 2, 3, 4))
            choose_thal = st.radio("your thal", ("normal", "fixed defect", "reversable defect"))
            if choose_thal == "normal":
                thal = 0
            if choose_thal == "fixed defect":
                thal = 1
            if choose_thal == "reversable defect":
                thal = 2
            if st.button("predict"):
                list = np.array([age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
                re = knnmodel_1.predict(pd.DataFrame(list))
                if re == 1: 
                    st.write("be careful, you have heart disease")
                if re == 0:
                    st.write("whoo... relief, you do not have heart disease")





