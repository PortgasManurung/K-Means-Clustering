import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


#ambil dataset
dataset = pd.read_csv("dataset.csv")
dataset.head()


#Menentukan variabel yang akan di klusterkan 
dataset_x = dataset.iloc[:, 1:3]
dataset_x.head()
#print(dataset_x)

#Visual persebaran data 
#plt.scatter(dataset.distance, dataset.speed, s =10, c = "c", marker = "o", alpha = 1)
#plt.show()

#Mengubah Variabel Data Frame Menjadi Array 
x_array =  np.array(dataset_x)
#print(x_array)

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
x_scaled

#Menentukan dan mengkonfigurasi fungsi kmeans
kmeans = KMeans(n_clusters = 3, random_state=123)


#Menentukan kluster dari data
kmeans.fit(x_scaled)


#Pusat cluster
print(kmeans.cluster_centers_)


#Menampilkan Hasil Kluster 
#print(kmeans.labels_)

#Menambahkan Kolom "kluster" Dalam Data Frame Driver
dataset["kluster"] = kmeans.labels_


#visual hasil kluster 
output = plt.scatter(x_scaled[:,0], x_scaled[:,1], s = 100, c = dataset.kluster, marker = "o", alpha = 1, )
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, alpha=1 , marker="s");
plt.title("K-Means Clustering")
plt.colorbar (output)
plt.show()