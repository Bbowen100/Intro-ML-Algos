# Intro-ML-Algos
<i>Introductory machine learning algorithms implemented from scratch with matlab.</i>

## Clustering
 ![Original Image](/imgs/GMM1.png?raw=true)
 
 <b>GMM</b> : Gaussian Mixture Models
 - A clustering algorithm that generates a set of gaussian distributions based on the training data and then uses the  max probability given by the distribution to predict which model generated the data. Below are pictures of the GMM algorithm being applied to an image of a flower. The GMM in this instance was tasked in finding 5 clusters for the data and cluster centers were picked at random from the dataset.
![GMM](/imgs/gmm_5_clusters.png?raw=true)

<b>KMEANS</b> : K Means Clustering
 - A clustering algorith that picks K data points at random, uses those data points as centers and assigns each data point a center by finding the center with the min L2 distance. Below is an imagage of the results of KMEANS finding 5 clusters in the flower image.
![KMEANS](/imgs/kmeans_5_clusters.png?raw=true)

<b>PCA</b> : Principle Component Analysis
 - PCA is traditionally used to generate lower dimensional data. It can also be used to compress data and reduce noise in the data. PCA was used to reduce the dimensionality of the image from 3 to 2. The resulting data was then reconstructed from 2 to 3 and this image is the result of the reconstruction.
 ![PCA](/imgs/pcs_reconstruction.png?raw=true)
