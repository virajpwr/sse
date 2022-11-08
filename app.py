from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('SSE Task')
st.subheader(
    'The task is to explore the data to reveal any seasonality or trends')

st.subheader(
    '**Please select the options on the sidebar to see different analysis performed on the data**')
genre = st.sidebar.radio(
    "Please select type",
    ('Statistics', 'Trend', 'outliers and transformation'))

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
    st.write("""
    - Shapiro-Wilk test result show that Var A is not normally distributed and Var B is normally distributed. 
    """)
    image_ci = Image.open(
        './plots/descriptive/gaussian.jpg')
    st.image(image_ci, width=450,
             caption='Shapiro-Wilk test result')

    st.markdown('**There are no missing values in the dataset.**')
    st.write("""
    Distribution of population
    - The skewness suggest that Var A and Var B are not skewed.
    - Kurtosis of Var A value 4.32 > 3 suggest there are outliers.
    - Kurtosis of Var B suggest negative kurtosis.
    """)
    image_stats = Image.open(
        './plots/descriptive/skewness.jpg')
    st.image(image_stats, width=250, caption='distribution of Var A and Var B')

    st.text('----------------------- Confidence interval of Var A and Var B------------------------')
    st.write("""
    - The 95% confidence interval of Var A and Var B is [797.4, 831.4] and [871.45, 913.02] respectively. 
    - The 95% CI was calculated on bootstrap samples of 1000 samples using sampling with replacement technique since there are outliers in Var A and Var B. 
    - Since the CI of Var A and Var B does not overlap, we can conclude that the mean of Var A and Var B are significantly different.
    """)
    # insert image
    image_ci = Image.open(
        './plots/descriptive/ci_vara.jpg')
    st.image(image_ci, width=500,
             caption='Confidence interval of Var A')

    image_ci = Image.open(
        './plots/descriptive/ci_varb.jpg')
    st.image(image_ci, width=500,
             caption='Confidence interval of Var A')

    st.text('----------------------- statistical tests---------------------------------------------')
    st.markdown('Independent two sample t-test')
    st.write("Performed independent two sample t-test to check if the means of Var A and Var B are significantly different with alpa = 0.05. The p-value is 0.000 which is less than 0.05. Hence, we reject the null hypothesis and conclude that the means of Var A and Var B are significantly different.")
    image_stats = Image.open(
        './plots/descriptive/ttest.jpg')
    st.image(image_stats, width=600, caption='t-test')
    st.markdown('ANOVA')
    st.write("Performed one way ANOVA to see if there is significant difference in year and month for variable Var A and Var B with alpha=0.05.")
    st.write('From the test result of One way ANOVA test shown below, there is no significant difference in the means of Var A for the years 2010 to 2019. However, there is a significant difference in the means of Var A for the months Jan to Dec.')
    image_stats = Image.open(
        './plots/descriptive/anova.jpg')
    st.image(image_stats, width=600, caption='ANOVA')
    st.write('There is significant difference in the means of Var B for the years 2010 to 2019 and for the months Jan to Dec.')
    image_stats = Image.open(
        './plots/descriptive/anova2.jpg')
    st.image(image_stats, width=600, caption='ANOVA')

if genre == 'Trend':
    st.header('Trend of Var A and Var B')
    st.text('----------------------------------------- Box Plots---------------------------------------------')
    st.markdown('Box plots of Var A and Var B for year')
    st.write("""
    - The distribution Var A and Var B in different years shows there are outliers in year 2018 and 2019 marked by 'x' in the box plots shown below.
    - The data points are considered outlier when a data lies outside of 𝜇±2⋅7𝜎.
    - Var A: The distribution has similar median across the years. The Interquartile range are similar, and the distribution are similar across the years with little to no skewness. The minimum value of Var A in year 2015 is much lower compared to other years and is left skewed. 
    - Var B: The distribution has similar median from year  2012 to 2017. However, for year 2010,2011,2018, and 2019 the median is significantly different to years from 2012 to 2017. The distribution is right skewed in year 2019 and left skewed in 2015.
    """)
    image_stats = Image.open('./plots/descriptive/box_1.jpg')
    st.image(image_stats, width=750, caption='Box plots of Var A and Var B')
    st.markdown('Box plots of Var A and Var B for Month')
    st.write("""
    - The box plot shows distribution of Var A and Var B over different months.
    - The distribution shows there are outliers in the month of  Apr, June, July, August, Sep and Nov in Var A and Feb, June, July, August Var B.
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
    - The means of Var B is significantly higher than means of Var A.
    - The var B sees a downward trend till 2015 and an upward trend from 2015 to 2019.
     Medians over Months:
    - Median is chosen to because of non-symmetric distribution of Var A an Var B in different months
    - The median of Var A and Var B is lowest in the month of June and highest in December.
    - Both variable show an downward trend from Jan to June and an upward trend from June to December.
    """)
    image_stats = Image.open('./plots/descriptive/trend_1.jpg')
    st.image(image_stats, width=750, caption='Line plots of Var A and Var B')

    st.markdown('Trend of Var A and Var B over year for different months')
    st.write("""
    - The trend line shows trend of var A and var B from  2010 to 2019.   
    - We can see the two outliers present in 2018 for Var A and Var B in the month of June and July.
    - We can see that in 2015 the minimum for var a and var b was in August.
     """)
    image_stats = Image.open('./plots/descriptive/trend_2.jpg')
    st.image(image_stats, width=750, caption='Trend of Var A and Var B')

    st.markdown('Trend of Var A and Var B over different years with CI')
    st.write("""
    - The line plot shows the trend of Var A and Var B with respect to year with blue line as mean value and the shaded blue area is 95% confidence interval. 
    - We can see the due to outliers the confidence interval is very wide for the year 2018 and 2019 for Var A and Var B compared to other years.
    """)

    image_stats = Image.open('./plots/descriptive/line_year.jpg')
    st.image(image_stats, width=750,
             caption='Trend of Var A and Var B over year and CI')

    st.markdown('Trend of Var A and Var B over different months with CI')
    st.write("""
    - The line plot shows the trend of Var A and Var B with respect to month with blue line as mean value and the shaded area as 95% confidence interval.    
    - Similar to the line plot above we can se that due to outliers the confidence interval is wider for the month of June, July and August for Var A and Var B compared to other months.
    """)

    image_stats = Image.open('./plots/descriptive/line_month.jpg')
    st.image(image_stats, width=750,
             caption='Trend of Var A and Var B over year and CI')

    st.text('-------------------------------------Relation between continous variables---------------------------------------------')
    st.write("""
    - The scatter plot shows linear relationship between Var A and Var B  with a correlation coefficient of 0.8.
    - The outliers can be seen at the opposite end of x axis for year 2018 and 2019.
    """)
    image_stats = Image.open('./plots/descriptive/corr.jpg')
    st.image(image_stats, width=850, caption='Relation between Var A and Var B')

if genre == 'outliers and transformation':
    st.text('------------------------------------- Outliers ---------------------------------------------')
    # st.markdown('Outliers in Var A and Var B')
    # st.write("""
    # - The box plot shows the outliers present in Var A and Var B. """)
    # image_stats = Image.open('./plots/descriptive/out_vara.jpg')
    # st.image(image_stats, width=500, caption='Outliers in Var A and Var B')
    # image_stats = Image.open('./plots/descriptive/out_var_a_b.jpg')
    # st.image(image_stats, width=500, caption='Distribution of  Var A and Var B')

    st.markdown('Outliers in Var A and Var B grouped by year and month')
    st.write("""
    - Bar plot below shows the outliers in Var A and Var B grouped by year and month. The data points are considered outlier when a data lies outside of 𝜇±2⋅7𝜎. 
    - The Var A and Var B have same number of outliers in year 2018 and 2019.
    - Var A has more outliers than Var B for different months when the outliers are found in each month.
    - Both variables have outliers in the month of Jun, Jul, Aug, Sep. 
    """)
    image_stats = Image.open('./plots/descriptive/outliers.jpg')
    st.image(image_stats, width=750, caption='Outliers in Var A and Var B')

    st.write(""" To deal with outliers we can use different methods like:
    - Removing outliers 
    - Log transformation of Var A and Var B
    - Imputing outliers with median
    """)
    st.text('------------------------------------- removing outliers ---------------------------------------------')
    # st.markdown('Removing outliers in Var A and Var B')
    # st.write("""
    # - Distribution of Var A and Var B after removing outliers. """)
    # image_stats = Image.open('./plots/descriptive/dist_not_outliers.jpg')
    # st.image(image_stats, width=650, caption='Distribution of Var A and Var B after removing outliers')

    st.markdown('Removing outliers in Var A and Var B over year')
    st.write("""
    - From the line plot below we can see the trend of Var A and Var b after removing the outliers.
    - The mean of  Var A and Var B and the confidence interval for year 2018 and 2019 changes and we can see that there is no data for the month of july for both years.""")
    image_stats = Image.open('./plots/descriptive/outliers_removed.jpg')
    st.image(image_stats, width=750,
             caption='Removing outliers in Var A and Var B')

    image_stats = Image.open('./plots/descriptive/outliers_removed_ci.jpg')
    st.image(image_stats, width=750,
             caption='Removing outliers in Var A and Var B')

    st.text('------------------------------------- Log transformation ---------------------------------------------')
    st.markdown('Log transformation of Var A and Var B')
    st.write("""
    - Log transformation is performed to reduce the variation caused by the outliers.
    - Log transformation de-emphasizes the extreme values and makes the distribution more normal.
    - The trend of log transformed Var A and Var B is similar to the original data. However the influence of outliers is reduced.
    """)
    image_stats = Image.open('./plots/descriptive/trend_log.jpg')
    st.image(image_stats, width=700,
             caption='Trend of log tranformed Var A and Var B')
    image_stats = Image.open('./plots/descriptive/vara_log.jpg')
    st.image(image_stats, width=700,
             caption='Distributio log tranformed Var A and Var A')
    image_stats = Image.open('./plots/descriptive/varb_log.jpg')
    st.image(image_stats, width=700,
             caption='Distributio log tranformed Var B and Var B')
    st.text('------------------------------------- Imputing outliers with median ---------------------------------------------')
    st.markdown('Imputing outliers with median')
    st.write("""
    - The outliers are imputed grouped by median of the month.
    - By imputing the outliers with median the confidence interval for year 2018 and 2019 changes significantly.
    - The mean and confidence interval for month of June and July for Var A and Var B has changed significantly.
    - The 95% confidence interval for month of June and July for Var A and Var B has changed from the original data and is very narrow.
    - The 95% CI for Var A and Var B has changeed to [803.57, 829.14] and [868.59 905.02] respectively.
    - We do not lose any data points by imputing the outliers with median.
    """)
    image_stats = Image.open('./plots/imputed/ci_year.jpg')
    st.image(image_stats, width=750,
             caption='Trend of Var A and Var B with respect to year after imputation of outliers with median')

    image_stats = Image.open('./plots/imputed/ci_months.jpg')
    st.image(image_stats, width=750,
             caption='Trend of Var A and Var B with respect to year after imputation of outliers with median')

    image_stats = Image.open('./plots/imputed/mean_trend_year.jpg')
    st.image(image_stats, width=750,
             caption='Trend of Var A and Var B with respect to year after imputation of outliers with median')

    image_stats = Image.open('./plots/imputed/mean_trend_month.jpg')
    st.image(image_stats, width=750,
             caption='Trend of Var A and Var B with respect to month after imputation of outliers with median')

    image_stats = Image.open('./plots/imputed/distribution.jpg')
    st.image(image_stats, width=750,
             caption='Distirbution of Var A and Var B after imputation of outliers with median')
    st.markdown(
        'Scatter plot of Var A and Var B after imputation of outliers with median')
    st.write("""
    - The scatter plot below shows the relationship between Var A and Var B after imputation of outliers with median.
    - We can see heteroscedasticity in the scatter plot. The variance of Var B increases as the value of Var A increases.
    - The extreme outlliers are removed and the relationship between Var A and Var B is not affected.
    """)
    image_stats = Image.open('./plots/imputed/scatter.jpg')
    st.image(image_stats, width=750,
             caption='Scatter plot of Var A and Var B after imputation of outliers with median')

    st.write("""
    - I performed three methods to deal with outliers in Var A and Var B in the analysis.
    - Deletion of outliers - The outliers were removed which causes loss of data points and it changes the point estimates and confidence interval. There is loss of information in the data when performing analysis.
    - Log transformation - Log of Var A and Var B was taken to reduce the variation caused by the outliers. The trend of log transformed Var A and Var B is similar to the original data. However the influence of outliers is reduced.
    - Imputing outliers with median - The outliers are imputed grouped by median of the month. There is no loss of data by imputation.
    """)
