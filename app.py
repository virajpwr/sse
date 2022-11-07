from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('SSE Task')
st.subheader('The task is to perform EDA on the data provided.')
st.subheader('Please select the options in the sidebar to see the EDA.')
genre = st.sidebar.radio(
    "Please select type",
    ('Statistics', 'Trend'))

# chart_select = st.sidebar.selectbox(
#     label ="Type of chart",
#     options=['Descriptive statistics','box plots','Trend']
# )
# page = st.sidebar.selectbox("Choose a page", ['Exploration', 'Prediction'])

if genre == 'Statistics':
    st.header('Statistics')
    st.subheader('The dataset contains four columns: **Year**, **Month** and **Var A** and **Var B** and 120 rows. The **Year** column ranges from 2010 to 2019 and the **Month** column ranges from Jan to Dec.')
    st.markdown('Dataset')
    image_data = Image.open('./plots/descriptive/dataset.jpg')
    st.image(image_data, width=250, caption='Dataset')

    st.text('----------------------- Descriptive statistics of the dataset------------------------**')
    st.markdown('The image below shows the descriptive statistics of the Var A and Var B. The **mean** and **Std deviation** of Var B is higher than Var A. The **range** of Var A is 791 is higher than and Var B range which 786')
    # insert image
    image_stats = Image.open(
        './plots/descriptive/stats.jpg')
    st.image(image_stats, width=250,
             caption='Descriptive statistics of Var A and Var B')

    st.markdown('There are no missing values in the dataset.')
    st.write("""
    Distribution of population
    - The skewness suggest that Var A and Var B are not skewed.
    - Kurtosis of Var A value 4.32 > 3 suggest there are outliers.
    - Kurtosis of Var B suggest negative kurtosis.
    """)
    image_stats = Image.open(
        './plots/descriptive/skewness.jpg')
    st.image(image_stats, width=250, caption='distribution of Var A and Var B')
    st.text('----------------------- statistical tests------------------------**')
    st.markdown('Independent two sample t-test')
    st.write("Performed independent two sample t-test to check if the means of Var A and Var B are significantly different with alpa = 0.05. The p-value is 0.000 which is less than 0.05. Hence, we reject the null hypothesis and conclude that the means of Var A and Var B are significantly different.")
    image_stats = Image.open(
        './plots/descriptive/ttest.jpg')
    st.image(image_stats, width=600, caption='t-test')
    st.markdown('ANOVA')
    st.write("Performed one way ANOVA to check difference in year and month for variable Var A and Var B with alpha=0.05.")
    st.write('There is no significant difference in the means of Var A for the years 2010 to 2019. However, there is a significant difference in the means of Var A for the months Jan to Dec.')
    image_stats = Image.open(
        './plots/descriptive/anova.jpg')
    st.image(image_stats, width=600, caption='ANOVA')
    st.write('There is significant difference in the means of Var B for the years 2010 to 2019 and for the months Jan to Dec.')
    image_stats = Image.open(
        './plots/descriptive/anova2.jpg')
    st.image(image_stats, width=600, caption='ANOVA')

# if genre == 'Trend':
#     st.header('Box plots')
#     st.markdown('The box plots below show the distribution of Var A and Var B for the years 2010 to 2019 and for the months Jan to Dec.')
#     image_stats = Image.open(
#         r'./plots/boxplots/boxplot.jpg')
#     st.image(image_stats, width=600, caption='Box plots')
#     st.markdown('The box plots below show the distribution of Var A and Var B for the years 2010 to 2019 and for the months Jan to Dec.')
#     image_stats = Image.open(
#         r'./plots/boxplots/boxplot2.jpg')
#     st.image(image_stats, width=600, caption='Box plots')
