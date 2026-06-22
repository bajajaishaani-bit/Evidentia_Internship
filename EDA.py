import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('ggplot')
pd.set_option('display.max_columns', 30)
#pd.set_option('display.max_rows', 266)
df = pd.read_csv('/Users/aishaanibajaj/Downloads/OOP_cleaned.csv')
print(df.shape)
print(df.head)
print(df.dtypes)
print(df.describe)
print(df.columns)
df = df[['Country Name', 'Country Code','Indicator Code', 'Indicator Name',
      '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
       '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
       '2018', '2019', '2020', '2021', '2022', '2023', '2024', 'has_data'
 ]].copy()
df.drop(['Country Code', 'Indicator Code'], axis=1, inplace=True)
print(df.dtypes)
print(df.head)
#renaming columns
df = df.rename(columns={'Indicator Name': 'Metric',})
print(df.head)
print(df.isna().sum())
print(df.duplicated().sum())
#box plot for 2020
plt.figure(figsize=(10, 6))
df['2020'].dropna().plot(kind='bar')
plt.ylabel('Out-of-pocket expenditure (% of health spending)')
plt.xlabel('Country')
plt.title('Distribution of Out-of-pocket Expenditure Across Countries (2020)')
plt.show()
#feature relationships
df.plot(kind='scatter', x = '2020', y = '2024')
plt.show()
sns.scatterplot(x = '2020', y = '2024', hue = 'has_data', data = df)
plt.show()
#questions about the data
#which metric has the highest value
df.head()
print(df['Metric'].value_counts())
print(df.query('Metric == "Out-of-pocket expenditure"').groupby('Metric').agg(['count']))


