import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('audi.csv')

# Clean price column
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Add age of the car
df['age'] = 2025 - df['year']

# Correlation matrix
corltn_matrix = df[['price', 'mileage', 'tax', 'mpg', 'engineSize', 'age']].corr()

# Plotting
plt.figure(figsize=(14, 10))

# 1. Price Distribution
plt.subplot(2, 2, 1)
sns.histplot(df['price'], bins=30, kde=True, color='skyblue')
plt.title('Price Distribution')
plt.xlabel('Price')

# 2. Average Price by Transmission Type
plt.subplot(2, 2, 2)
sns.barplot(x='transmission', y='price', hue='transmission', data=df, palette='Set2', legend=False)

plt.title('Average Price by Transmission Type')
plt.ylabel('Avg Price')

# 3. Engine Size vs Price
plt.subplot(2, 2, 3)
sns.scatterplot(x='engineSize', y='price', data=df, hue='fuelType')
plt.title('Engine Size vs Price')
plt.xlabel('Engine Size (L)')
plt.ylabel('Price')

# 4. Correlation Matrix Heatmap
plt.subplot(2, 2, 4)
sns.heatmap(corltn_matrix, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Correlation between Features')

# Prevent overlapping
plt.tight_layout()
plt.savefig('audi_car_analysis_visulization.png')
plt.show()
