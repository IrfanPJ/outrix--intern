import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('StudentsPerformance.csv')

print(df.head())
df= df.dropna()

print(df[['math score', 'reading score', 'writing score']].describe())

plt.figure(figsize=(8,5))
sns.histplot(df['math score'], kde=True, bins=20)
plt.title('Distribution of Math Scores')
plt.xlabel('Math Score')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='gender', y='math score', data=df)
plt.title('Math Score Distribution by Gender')
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df[['math score', 'reading score', 'writing score']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Scores')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='test preparation course', y='math score', data=df)
plt.title('Average Math Scores by Test Preparation Course')
plt.xticks(rotation=45)
plt.show()
