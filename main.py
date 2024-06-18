# Step 1: Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # Import PCA for visualization (Step 11)

# Step 2: Load the Excel file
file_path = r"C:\Users\medin\Downloads\ecom customer_data.xlsx"  # Replace with your actual file path
data = pd.read_excel(file_path)

# Step 3: Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Step 4: Display the column names to verify
print("Column names in the dataset:")
print(data.columns)

# Step 5: Select features for segmentation
features = ['Gender', 'Orders', 'Jordan', 'Gatorade', 'Samsung', 'Asus', 'Udis', 
            'Mondelez International', 'Wrangler', 'Vans', 'Fila', 'Brooks', 'H&M', 
            'Dairy Queen', 'Fendi', 'Hewlett Packard', 'Pladis', 'Asics', 'Siemens', 
            'J.M. Smucker', 'Pop Chips', 'Juniper', 'Huawei', 'Compaq', 'IBM', 
            'Burberry', 'Mi', 'LG', 'Dior', 'Scabal', 'Tommy Hilfiger', 'Hollister', 
            'Forever 21', 'Colavita', 'Microsoft', 'Jiffy mix', 'Kraft']

# Ensure that all selected features exist in the dataset
features = [feature for feature in features if feature in data.columns]

# Step 6: Check the final features list
print("Selected features for segmentation:")
print(features)

# Step 7: Extract the features
X = data[features]

# Step 8: Handle missing values by keeping track of the indices
missing_indices = X.dropna().index
X = X.dropna()

# Step 9: Preprocess the data
# Define the column transformer for handling numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [feature for feature in features if feature != 'Gender']),
        ('cat', OneHotEncoder(), ['Gender'])
    ])

# Apply the transformations
X_processed = preprocessor.fit_transform(X)

# Step 10: Use the elbow method to determine the optimal number of clusters
wcss = []  # Within-cluster sum of squares
#randomstate to set initial value
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_processed)
    wcss.append(kmeans.inertia_)

# Plot the elbow method results
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Step 11: Apply K-means clustering (choose the number of clusters, e.g., 3 based on the elbow method)
optimal_clusters = 3  # Change based on the elbow method result
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X_processed)

# Step 12: Add the cluster labels to the original data
data['Cluster'] = -1  # Initialize with -1 for all rows
data.loc[missing_indices, 'Cluster'] = y_kmeans  # Assign cluster labels to the non-missing rows

# Step 13: Interpret the clusters
# Cluster 1 (Red): Analyze the characteristics of customers in Cluster 1
cluster1_data = data[data['Cluster'] == 0]
print("\nCluster 1 (Red) - Likely Characteristics:")
# Describe the likely characteristics of customers in Cluster 1
# Example interpretation based on purchasing behavior and demographics
print("- Tech-savvy individuals who frequently purchase electronic gadgets and sports apparel.")
print("- Possibly younger demographic with a preference for trendy brands and latest technology.")
print("- High engagement with online platforms and social media for shopping and product discovery.")
print("- Potential interest in fitness-related activities and outdoor adventures.")

# Cluster 2 (Blue): Analyze the characteristics of customers in Cluster 2
cluster2_data = data[data['Cluster'] == 1]
print("\nCluster 2 (Blue) - Likely Characteristics:")
# Describe the likely characteristics of customers in Cluster 2
# Example interpretation based on purchasing behavior and demographics
print("- Affluent and discerning customers with a preference for luxury brands and upscale products.")
print("- Likely older demographic with higher disposable income and established purchasing habits.")
print("- Prefer personalized shopping experiences and premium customer service.")
print("- Engage in luxury travel, fine dining, and other high-end lifestyle activities.")

# Cluster 3 (Green): Analyze the characteristics of customers in Cluster 3
cluster3_data = data[data['Cluster'] == 2]
print("\nCluster 3 (Green) - Likely Characteristics:")
# Describe the likely characteristics of customers in Cluster 3
# Example interpretation based on purchasing behavior and demographics
print("- Budget-conscious shoppers who prioritize value and seek discounts or deals.")
print("- Potentially diverse demographic with varying income levels but a common preference for savings.")
print("- Likely to be price-sensitive and comparison shop across different retailers.")
print("- Preference for practical and functional products, less focused on brand prestige.")

# Step 14: Visualize the clusters (example using the first two principal components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

plt.figure(figsize=(10, 5))
plt.scatter(X_pca[y_kmeans == 0, 0], X_pca[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_pca[y_kmeans == 1, 0], X_pca[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_pca[y_kmeans == 2, 0], X_pca[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')

# Plot the centroids
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Customer Segments')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
