from data_gen import Feature_Extraction
from train_utils import GAN,train

real_size = (32,32,3)
z_size = 100
learning_rate = 0.001
batch_size = 128
epochs = 25
proj_directory = '/Users/Theologis/Desktop/CAS Lab/chars74k/'

net = GAN(real_size, z_size, learning_rate)

dataset = Feature_Extraction(minibatch_size=batch_size,proj_directory= proj_directory,real_size=real_size )

train_accuracies, test_accuracies, samples = train(net, dataset, epochs, batch_size, figsize=(10,5))