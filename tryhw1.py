from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#
# For each layer, the output volume's spatial size is: (W-K+2P)/S + 1
# Where W is the input volume, K is the kernel size, P is the padding amount,
# and S is the stride.
#
# For example, for the first conv layer, the input is 32*32*3 with 8 filters, each 3*3 with a stride of 1.
# Thus, the output size is (32-3+2(1))/1 + 1 = 32.
# Knowing this, the number of operations is follows: 32*32*3*3*3*8 = 221,184
# 
# 1st Conv layer: 221,184 (from example above)
# 1st Maxpool layer: there's no multiplicaton operations in maxpool layers so ignore these
# 2nd Conv layer: 16*16*3*3*8*16 = 294,912
# 2nd Maxpool layer: there's no multiplicaton operations in maxpool layers so ignore these
# 3rd Conv layer: 8*8*3*3*16*32 = 294,912
# 3rd Maxpool layer: there's no multiplicaton operations in maxpool layers so ignore these
# 4th Conv layer: 4*4*3*3*32*64 = 294,912
# 4th Maxpool layer: there's no multiplicaton operations in maxpool layers so ignore these
# Fully connected layer: 256*10 = 2,560
#
# Total # of operations: 1,108,480
