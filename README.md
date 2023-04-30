# Image segmentation using K-means and EM

 Image Segmentation Using K-Means Clustering, Gaussian Mixture Model and Expectation Maximization

K-means clustering can be used to extract related pixels from the remainder of the image and group them together for the purpose of image segmentation. The extracted features from the image, object classification, or additional analysis can all be done using the generated clusters. This algorithm works by first randomly initializing K cluster centroids, which act as representatives of each cluster. It then assigns each data point to the closest centroid based on a distance metric such as Euclidean distance. After all the data points have been assigned to a centroid, the algorithm recalculates the centroid of each cluster as the mean of all the data points assigned to that cluster. This process of reassignment and centroid update is repeated until the centroids no longer move or the maximum number of iterations is reached.
The Expectation-Maximization (EM) algorithm is a widely used statistical method for estimating parameters of probabilistic models, particularly in situations where some of the data is missing or unobserved. One important application of the EM algorithm is in fitting Gaussian mixture data, where the data is assumed to be generated from a mixture of one or more Gaussian distributions. This type of model is useful in many areas, such as computer vision, image processing, and machine learning, where it is often necessary to identify and separate different types of objects or features in images. The EM algorithm is an iterative procedure that alternates between estimating the unknown parameters of the model and computing the expected values of the missing or unobserved data, given the current estimates of the parameters. This process continues until convergence is achieved, and the final estimates of the parameters are used to describe the underlying distribution of the data. In this way, the EM algorithm provides a powerful tool for modeling complex data and extracting meaningful information from it.

K-means algorithm:

<img width="630" alt="image" src="https://user-images.githubusercontent.com/36191021/235366944-76762400-88ff-431e-80b0-ec20ae1e85eb.png">

Expectation maximization algorithm:

1. Initialize 𝜋𝑘, 𝝁𝑘, 𝜮𝑘, 𝑘 = 1, 2, ... , 𝐾, and evaluate the initial value of the log likelihood.

2. Expectation step of the Exepectation Maximization (EM) process:
<img width="630" alt="image" src="https://user-images.githubusercontent.com/36191021/235366672-00268a2e-1d2c-424f-b2b9-36cab8c85bd9.png">

3. Maximization step of the Expectation Maximization (EM) process:
<img width="630" alt="image" src="https://user-images.githubusercontent.com/36191021/235366641-3247b143-2bff-4d37-a736-3c69e39c98d3.png">

4. Calculate and update the log-likelihood:
<img width="630" alt="image" src="https://user-images.githubusercontent.com/36191021/235366785-079c5cf1-e16f-4b43-9e52-866d2d194992.png">

