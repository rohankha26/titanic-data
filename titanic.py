# import seaborn as sns
# import pandas as pd
# import streamlit as st
# import numpy
# import matplotlib.pyplot as plt

# df = sns.load_dataset('titanic')
# df.head()
# df_clean = df.dropna(subset=['age', 'embarked'])


# # df_clean = df.dropna(subset=['age', 'embarked'])
# # print(f"Original: {len(df)} passengers")
# # print(f"After cleaning: {len(df_clean)} passengers")
# # plt.figure(figsize=(8, 5))

# # Chart 3: Box Plot
# # st.header("3. Tips by Day")
# # fig3, ax3 = plt.subplots()
# # sns.boxplot(data=df, x='day', y='tip', ax=ax3)
# # st.pyplot(fig3)
# st.header('Survival Count by Passenger Class')
# fig1, ax1 = plt.subplots()
# sns.countplot(data=df_clean, x='class', hue='survived')
# st.pyplot(fig1)

# # sns.countplot(data=df_clean, x='class', hue='survived')
# # plt.title('Survival Count by Passenger Class')
# st.header('Age Distribution: Survivors vs Non-Survivors')
# fig2,ax2 =plt.subplot()
# sns.histplot(data=df_clean, x='age', hue='survived', bins=30, kde=True)
# st.pyplot(fig2)

# st.header('Survival Rate by Gender (%)')

# avg_fare_by_class = df_clean.groupby('class')['fare'].mean()

# # st.header('Average Fare Paid by Class')
# # fig4, ax4 = plt.subplots()
# # avg_fare_by_class.plot(kind='bar', color=['#F18F01', '#C73E1D', '#6A994E'])
# # st.pyplot(fig4)
# # fig3, ax3 = plt.subplots()
# # sns.barplot(x=survival_by_gender.index, y=survival_by_gender.values)
# # st.pyplot(fig3)


# # plt.figure(figsize=(8, 5))
# # sns.histplot(data=df_clean, x='age', hue='survived', bins=30, kde=True)
# # plt.title('Age Distribution: Survivors vs Non-Survivors')

# # survival_by_gender = df_clean.groupby('sex')['survived'].mean() * 100
# # survival_by_gender

# # plt.figure(figsize=(8, 5))
# # sns.barplot(x=survival_by_gender.index, y=survival_by_gender.values)
# # plt.title('Survival Rate by Gender (%)')
# # plt.ylabel('Survival Rate (%)')

# # avg_fare_by_class = df_clean.groupby('class')['fare'].mean()
# # avg_fare_by_class

# # plt.figure(figsize=(8, 5))
# # avg_fare_by_class.plot(kind='bar', color=['#F18F01', '#C73E1D', '#6A994E'])
# # plt.title('Average Fare Paid by Class')
# # plt.ylabel('Average Fare ($)')


import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')


df_clean = df.dropna(subset=['age', 'embarked'])


st.header('Survival Count by Passenger Class')
fig1, ax1 = plt.subplots()
sns.countplot(data=df_clean, x='class', hue='survived')
st.pyplot(fig1)

st.header('Age Distribution: Survivors vs Non-Survivors')
fig2, ax2 = plt.subplots()
sns.histplot(data=df_clean, x='age', hue='survived', bins=30, kde=True)
st.pyplot(fig2)

survival_by_gender = df_clean.groupby('sex')['survived'].mean() * 100

st.header('Survival Rate by Gender (%)')
fig3, ax3 = plt.subplots()
sns.barplot(x=survival_by_gender.index, y=survival_by_gender.values)
st.pyplot(fig3)



avg_fare_by_class = df_clean.groupby('class')['fare'].mean()

st.header('Average Fare Paid by Class')
fig4, ax4 = plt.subplots()
avg_fare_by_class.plot(kind='bar', color=['#F18F01', '#C73E1D', '#6A994E'])
st.pyplot(fig4)