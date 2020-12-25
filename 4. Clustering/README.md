# k-means Clustering
## Problem Setup
The EMNIST’s letters database is a dataset of handwritten letters, comprising 124800 training
examples and 20800 test examples. In this question, we will implement the k-means algorithm using EMNIST’s letters dataset. The data can be downloaded at begining page. The EMNIST letters data set contains 26 various letters(with both uppercase and lowercase). Each of the letters is a 28x28 pixel image, resulting in a 784-dimensional space. The training set contains two files:
- *emnist-letters-train-images-idx3-ubyte :* training set images (97843216 bytes)
  
  *Contains the training image instances. Rows are images and columns are pixels with values from 0 to 255.*

- *emnist-letters-train-labels-idx1-ubyte :* training set labels (124808 bytes)
  
  *Contains the true labels of training images.*

And the testing set contains the following 2 files:
- *emnist-letters-test-images-idx3-ubyte :* testing set images (16307216 bytes)
  
  *Contains the testing image instances.*

- *emnist-letters-test-labels-idx1-ubyte :* testing set labels (20808 bytes)
  
  *Contains the true labels of testing images.*

Note that they are binary files and the dataset structure matches the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). You can get detailed information of the data from [here](
https://www.nist.gov/itl/products-and-services/emnist-dataset/) and the [paper](https://arxiv.org/pdf/1702.05373v1.pdf).
## Decode ubyte File to txt File
```python
import numpy as np
import struct
import random

train_images_idx3_ubyte_file = './emnist-letters-train-images-idx3-ubyte'
train_labels_idx1_ubyte_file = './emnist-letters-train-labels-idx1-ubyte'

test_images_idx3_ubyte_file = './emnist-letters-test-images-idx3-ubyte'
test_labels_idx1_ubyte_file = './emnist-letters-test-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):

    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic number:%d, image #: %d, image size: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('Finish %d' % (i + 1))
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    
    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic number:%d, image #: %d' % (magic_number, num_images))

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('Finish %d' % (i + 1))
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()

train_images_txt = []
for i in train_images:
    train_images_txt.append(i.flatten())
train_labels_txt = []
for i in train_labels:
    train_labels_txt.append(i.flatten())
test_images_txt = []
for i in test_images:
    test_images_txt.append(i.flatten())
test_labels_txt = []
for i in test_labels:
    test_labels_txt.append(i.flatten())

centroid = []
for i in range(38):
    centroid.append(random.randint(0,len(train_images_txt)))

with open("train_image.txt","w") as f:
    for i in range(len(train_images_txt)):
        data = str(list(train_images_txt[i])).strip('[').strip(']').replace(',','').replace('\'','')+'\n'
        f.write(data)
        if i in centroid:
            with open("random_centroid.txt","a") as g:
                g.write(data)
print("train image saved in train_image.txt ...... success!")
with open("train_label.txt","w") as f:
    for i in range(len(train_labels_txt)):
        data = str(list(train_labels_txt[i])).strip('[').strip(']').replace(',','').replace('\'','')+'\n'
        f.write(data)
print("train label saved in train_label.txt ...... success!")
with open("test_image.txt","w") as f:
    for i in range(len(test_images_txt)):
        data = str(list(test_images_txt[i])).strip('[').strip(']').replace(',','').replace('\'','')+'\n'
        f.write(data)
print("test image saved in test_image.txt ...... success!")
with open("test_label.txt","w") as f:
    for i in range(len(test_labels_txt)):
        data = str(list(test_labels_txt[i])).strip('[').strip(']').replace(',','').replace('\'','')+'\n'
        f.write(data)
print("test label saved in test_label.txt ...... success!")
```
## Choose Suitable k
Before using k-means, we first need to determine the key parameter k, a.k.a, the number of clusters. Since we have 26 different labels corresponding to the 26 letters and every letter has uppercase and lowercase, the k parameter should be an integer between 26 and 52. Notice that some letter have similar uppercase and lowercase.The alphabets with ※ size means that their uppercases and lowercases are similar. 
![](pic/cluster,jpg)\

Consequently, we consider 52-12=40 clusters