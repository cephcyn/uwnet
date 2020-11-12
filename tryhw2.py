from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            #make_batchnorm_layer(10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 100  # changed from 500 to 100 because we have 5 rounds of annealing eta
rate = .1
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
train_image_classifier(m, train, batch, iters, 0.08, momentum, decay)  # decrease learning rate
train_image_classifier(m, train, batch, iters, 0.04, momentum, decay)  # decrease learning rate
train_image_classifier(m, train, batch, iters, 0.02, momentum, decay)  # decrease learning rate
train_image_classifier(m, train, batch, iters, 0.01, momentum, decay)  # decrease learning rate

print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# Section 7.6 Responses
#
# Training conv_net without batchnorm:
# Training accuracy = 0.3507
# Test accuracy = 0.3589
#
# Training conv_net with batchnorm before every activation:
# Training accuracy = 0.5318
# Test accuracy = 0.5182
#
# Training conv_net with batchnorm before every activation EXCEPT final Softmax:
# Training accuracy = 0.5309
# Test accuracy = 0.5244
#
# Compared to the convolutional neural network that had the same architecture but did not have
# batchnorm layers, the version with batchnorm performs better! We found that by adding a batchnorm
# layer right before every activation fucnction except the final softmax output yielded the greatest
# increase in performance. Specifically, the test accuracy increased by 0.1655 from 0.3589 to 0.5244.
#
# Annealing learning rate:
# Because we use batchnorm, our training process is more stable which allows us to use larger learning
# rates. By annealing the learning rate, we are able to achieve better convergence. We started with a
# learning rate of 0.1, and successively decreased it to 0.08, 0.05, and 0.01. We trained the model
# using the same values for the other hyperparameters.
#
# Training accuracy = 0.5976
# Test accuracy = 0.5735
#
# In order to test the effect of annealing the learning rate with the same total number of training
# iterations, we added one training round for a total of 5 rounds, with learning rate 0.1, 0.08, 0.04,
# 0.02, and 0.01, respectively. Each round had 100 iterations, for a total of 500 iterations (same as 
# before). As expected, we got slightly lower results, but they were still better than the network that
# did not utilize batchnorm, as well as the network with batchnorm before every activation layer.
#
# Training accuracy = 0.5233
# Test accuracy = 0.5155

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#

