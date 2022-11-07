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

    st.text('----------------------- Descriptive statistics of the dataset------------------------')
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
    st.text('----------------------- statistical tests---------------------------------------------')
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

if genre=='Trend':
    st.header('Trend of Var A and Var B')
    st.text('----------------------------------------- Box Plots---------------------------------------------')
    st.markdown('Box plots of Var A and Var B for year')
    st.write("""
    - The distribution Var A and Var B in different years shows there are outliers in year 2018 and 2019.
    - The data points are considered outlier when a data lies outside of 𝜇±2⋅7𝜎.
    - Var A: The distribution has similar median across the years. The Interquartile range are similar, and the distribution are similar across the years with little to no skewness. The minimum value of Var A in year 2015 is much lower compared to other years and is left skewed. 
    - Var B: The distribution has similar median from year  2012 to 2017. However, for year 2010,2011,2018, and 2019 the median is significantly different to years from 2012 to 2017. The distribution is right skewed in year 2019 and left skewed in 2015.

    """)
    image_stats = Image.open('./plots/descriptive/box_1.jpg')
    st.image(image_stats, width=750, caption='Box plots of Var A and Var B')
    st.markdown('Box plots of Var A and Var B for Month')
    st.write("""
    - The box plot shows distribution of Var A and Var B over different months.
    - The distribution shows there are outliers in the month of  Apr, June, July, August in Var A and Feb, June, July, August Var B.
    - The IQR and range for Var A is much smaller compared to IQR  and range of Var B.
    - Var B distributions are right skewed. While distribution of few months are skewed in Var A.
    - Both variable sees a trend of decreasing median by the month of June and increasing in the towards the month of December.

    """)
    image_stats = Image.open('./plots/descriptive/box_2.jpg')
    st.image(image_stats, width=750, caption='Box plots of Var A and Var B')

    st.text('----------------------------------------- Line Plots---------------------------------------------')
    st.markdown('Line plots of Var A and Var B for year')
    st.write("""
    Means over year:
    - The line plot show the trend of Var A mean and Var B median over Year and month
    - The mean Var A is the lowest in the year 2019 and highest in year 2018.
    - The means of Var A is significantly lower than means of Var B over the from 2015 to 2019.
    - The var B sees a downward trend till 2015 and an upward trend from 2015 to 2019.
     Medians over Months:
    - Median is chosen to because of non-symmetric distribution of Var A an Var B in different months
    - The median of Var A is Var B is lowest in the month of June and highest in December.
    - Both variable show an downward trend from Jan to June and an upward trend from June to December.
    """)
    image_stats = Image.open('./plots/descriptive/trend_1.jpg')
    st.image(image_stats, width=750, caption='Line plots of Var A and Var B')
    
    st.markdown('Trend of Var A and Var B over year for different months')
    st.write("""
    - The trend line shows trend of var a and var b from  201to 2019.   
    - We can see the two outliers present in 2018 for Var A and Var B in the month of June and July.
    - We can see that in 2015 the minimum for var a and var b was in August. """)
    image_stats = Image.open('./plots/descriptive/trend_2.jpg')
    st.image(image_stats, width=750, caption='Trend of Var A and Var B')

    st.text('-------------------------------------Relation between continous variables---------------------------------------------')
    st.write("""
    - The scatter plot shows linear relationship between Var A and Var B  with a correlation coefficient of 0.8.
    - The outliers can be seen at the opposite end of x axis for year 2018 and 2019.
    """)
    image_stats = Image.open('./plots/descriptive/corr.jpg')
    st.image(image_stats, width=850, caption='Relation between Var A and Var B')
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
