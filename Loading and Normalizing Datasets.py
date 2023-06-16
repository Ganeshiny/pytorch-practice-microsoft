
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

'''
Loading and normalizing datasets
'''

'''
parameters used and descriptions for them:

root = path where the test/train data is stored
train = specifies training or test dataset
download - True downloads the data from the internet, if its not available then the root
transform and target transform specify the feature and label transformations

'''
training_data = datasets.FashionMNIST(
    root = "data", 
    train= True,
    download = True, 
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root= "data", 
    train = False, 
    download=True, 
    transform= ToTensor()
)

'''
Interating and visualizing the dataset 

'''
#Indexing the datasets manually

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

#Using matplotlib to visualize some of the sample in the training data

#Creates a new figure object with a specified size of 8x8 inches.
figure = plt.figure(figsize=(8, 8)) 

cols, rows = 3, 3 #Sets the number of columns and rows in the grid to 3 each.

for i in range(1, cols * rows + 1): #Iterate through a range from 1 to 10 (3*3 + 1)
    sample_idx = torch.randint(len(training_data), size=(1,)).item() 
    #Generates a random index within the range of the length of the training_data dataset.
    #torch.randint returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive)
    #size variable arguement used here defines the shape of the output tensor
    #.item() returns the value of the tensor as a standard python number

    img, label = training_data[sample_idx]
    #Retrieves the image and its corresponding label from the dataset based on the random index generated.

    figure.add_subplot(rows, cols, i) #Adds a subplot to the figure in the grid specified by rows and cols, at the position i.

    plt.title(labels_map[label]) #Sets the title of the subplot to the label's name, using a dictionary called labels_map.
    plt.axis("off") #Turns off axis display for the subplot
    plt.imshow(img.squeeze(), cmap="gray") #Display the images in the subplot in a grayscale colormap
plt.show()

'''
preparing your data for training with DataLoaders

'''

from torch.utils.data import DataLoader

#DataLoader is an iterable that abstracts complexity of
#the dataset for use as easy API 


'''

data: the training data that will be used to train the model
this also test data to evaluate the model
batch_size: the number of models to be processed in each batch
The Dataset retreives the dataset's features one sample at a time,it passes the samples in 
'minibatches', reshuffle the data at every epoch to reduce the model overfitting. This is done 
using the Python's multiprocessing to speed up the data retrieval 

'''


train_dataloader = DataLoader(training_data, batch_size=64, shuffle= True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)

#Iterate through the DataLoader 


'''
Iterating through the DataLoader

'''

#Dataset stores the samples and their corresponding labels, 
#and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

#Display the image and the label
train_features, train_labels = next(iter(train_dataloader)) #iter method returns a iterator for an arguement and next returns the next item in an iterator
#It loads the first batch of data (features and labels) from the train_dataloader, which is an iterable object that provides batches of 
#training data. train_features contains the input data (e.g. images), and train_labels contains the corresponding labels (e.g. class labels).

img = train_features[0].squeeze() #It extracts the first feature (image) from the batch by taking the 0-th element of train_features 
#and removes any extra dimensions using the squeeze() method. This results in a 2D array img representing an image.

label = train_labels[0]
plt.imshow(img, cmap= "gray")
plt.show()
print(f"Label: {label}")


