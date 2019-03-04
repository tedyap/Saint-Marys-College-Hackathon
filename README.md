# Saint Mary's College Data Science Hackathon - Illinois Wesleyan University Data Science Club

[Hackathon Home Page](https://www.saintmarys.edu/math/data-science-hackathon)

# Data Science Hackathon Overview
We were given 16 hours to build a model to classify images into 40 different mathematical symbols. We built a simple Convolutional Neural Network and opimized accuracy with k-fold cross validation and majority voting ensemble learning. A more detailed description of our approach can be found in this blog post here. 

![alt text](

# Training Data
The training data comprises 99300 jpg images of handwritten mathematical symbols. Each image is greyscale and is
30x30. There are 40 different symbols. A symbol’s label is given in the file name. A file name is of the form
imageid_imagelabel.jpg, for example
32_infinity.jpg, or 33247_greater_than.jpg.

# Test Data
The only difference between an image in the test data set and the training data set is that there is no image label in the file name. So the name of a test image would be something like 45.jpg. There are 14000 test images.

