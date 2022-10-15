import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import hiplot as hip
import plotly.express as px
import altair as alt

# import dataset
heart_df = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/midterm_object/code/heart.csv")

# process data

target0 = heart_df.where(heart_df["target"] == 0).dropna()
target1 = heart_df.where(heart_df["target"] == 1).dropna()
names = list(heart_df.columns)
names.pop(-1)
ex_df = pd.DataFrame(index=["with heart diseases", "without heart disease"])
for i in names:
    ex_df[i] = [target1[i].mean(), target0[i].mean()]

# streamlit
## set container

data = st.container()
EDA = st.container()

with st.sidebar:
    st.title("EDA of Heart Diseases")
    add_radio = st.radio("NAVIGATOR", ("data introduction", "EDA"))

    if add_radio == "data introduction":
        with data:
            st.title("Introduction of Data")
            st.text("The heart disease dataset comes from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
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




    if add_radio == "EDA" :
        with EDA:
            st.header("Exploratory Data Analysis")
            add_selectbox = st.selectbox("choose one", ("hiplot", "relationship between attributes and heart diseases"))
            if add_selectbox == "hiplot":
                st.subheader("hipplot")
                hi = hip.Experiment.from_dataframe(heart_df)
                
                val = hi.to_streamlit( key="hip").display()

            if add_selectbox == "relationship between attributes and heart diseases":
                st.subheader("Relationship between Attributes and Heart Disease")
                # expected value table
                st.markdown("* the expected value of different attributes")
                st.write(ex_df)
                # correlation
                st.markdown("* the correlation plot")
                fig = plt.figure(figsize=(16, 16))
                corr = sns.heatmap(heart_df.corr(), annot=True)
                st.pyplot(fig)
                st.write("In the attributes with continuous value, old peak and thalach have high absolute \
                    correlations, age, trestbps and chol's correlations are relatively low. ")
                st.write("In the attibutes with discrete value, fbs only have two values, but the\
                     correlation with target is only -0.041, so I can confirm that fbs cannot help to predict the\
                        heart disease. Sex, cp, thal, exang, slope and ca have high absolute correlations \
                            with target, and restecg's absolute correlation is relatviely low.")
                
                # 2d_scatter plot about the old peak and thalch
                st.markdown("* the scatter plot about old peak and thalach")
                fig = alt.Chart(heart_df, title="the scatter plot between old peak and thalach").mark_point().encode(
                    x='thalach',
                    y='oldpeak',
                    color='target',
                    tooltip = ["thalach", "oldpeak", "target"]
                    ).interactive()
                st.altair_chart(fig)
                st.write("The plot shows the distribution of target=1 has difference from target=0's distribution. ")

                # 3d_scatter plot among old peak, thalch and exang
                st.markdown("* 3d_scatter plot among old peak, thalch and exang")
                st.write("Add another variable to the before plot and produce a 3-d scatter plot.")
                fig = px.scatter_3d(heart_df, x='thalach', y='oldpeak', z='exang', color='target')
                st.plotly_chart(fig)
                st.write("In this 3d-plot, the distribution of target=1 has greater difference \
                    from target=0's distribution. exang = 0 has higher proportion of target = 1 compared with exang = 1.")

                # discrete value
                st.markdown("* the distributions of attributes with discrete value")
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
                            the proportion of people with heart disease: 0.66; the expected cp value of people with heart disease is \
                                1.4; the expected cp value of people without heart disease is 0.48.")

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



















