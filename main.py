import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import package.accy as accy
import package.cost as cost
import package.data as data
import package.gradient as gradient

nargs = len(sys.argv)
message = ""
window_text=10
window_plot=16
epochs_max=5000 if nargs < 2 else int(sys.argv[1])
min_error=0.01 if nargs < 3 else float(sys.argv[2])
alpha=0.01 if nargs < 4 else float(sys.argv[3])

# Load dataset
trainLoad, testLoad, trackLoad, thetas = data.load()
train_set, train_y = trainLoad()
test_set, test_y = testLoad()

# Output sets
# train_accuracies = np.zeros([epochs_max])
train_errors = np.zeros([epochs_max])
train_predictions = np.zeros([train_y.size])

# test_accuracies = np.zeros([epochs_max])
test_errors = np.zeros([epochs_max])
test_predictions = np.zeros([test_y.size])

# Helper varianle
last_train_error = 0
last_test_error = 0
epoch=0

print("Max epochs", epochs_max, "-", "Min error", min_error, "-", "Learning Rate", alpha)

while(True):
    if(epoch >= epochs_max):
        message = "Max epochs reached"
        break
    # Gradient descent
    old_thetas = thetas
    thetas = gradient.descent(train_set, thetas, train_y, learning_rate=alpha)
    # Train
    train_predictions = np.dot(train_set, thetas)
    train_errors[epoch] = cost.rmse(train_y, train_predictions)
    # train_accuracies[epoch] = accy.r2(train_y, train_predictions)
    # Test
    test_predictions = np.dot(test_set, thetas)
    test_errors[epoch] = cost.rmse(test_y, test_predictions)
    # test_accuracies[epoch] = accy.r2(test_y, test_predictions)
    
    last_train_error = train_errors[epoch]
    last_test_error = test_errors[epoch]
    
    if(train_errors[epoch] < min_error):
        message = "Min error reached"
        break

    if((old_thetas==thetas).all()):
        message = "We are not improving anymore :("
        break
    epoch += 1
    

# Hypothesis vs real value
f, t, title = data.randomInterval(train_y, window_text, "Samples from", "to")
trained_y = train_y[f:t]
trained_p = train_predictions[f:t]
print(title)
for i, x in enumerate(trained_y):
    print('\thyp', trained_y[i], 'y', trained_p[i])

# Results
# train_r2 = accy.r2(train_y, train_predictions)
# test_r2 = accy.r2(test_y, test_predictions)

np.set_printoptions(precision=16)
print(message, "Epoch", str(epoch), "\n")
print("Train size", train_predictions.size)
print("Test size", test_predictions.size)
print("Train RMSE", last_train_error)
print("Test RMSE", last_test_error)
# print("Train R^2", train_r2)
# print("Test R^2", test_r2)
print("\nParams", thetas, "\n")

# Random tracks ids
data.randomTracks(trackLoad(), 2)

# Graphs
if(nargs < 5):
    legend = ['Song Hotttnesss', 'Hypothesis']

    # Plot results
    f = random.randint(0, train_y.size - window_plot - 2)
    t = f + window_plot
    title = "Train " + str(f) + "-" + str(t)
    plt.figure(1)
    plt.plot(train_y[f:t])
    plt.plot(train_predictions[f:t])
    plt.legend(legend)
    plt.title(title)

    f = random.randint(0, test_y.size - window_plot - 2)
    t = f + window_plot
    title = "Test " + str(f) + "-" + str(t)
    plt.figure(2)
    plt.plot(test_y[f:t])
    plt.plot(test_predictions[f:t])
    plt.legend(legend)
    plt.title(title)
    # plt.figure(4)
    # ff = np.empty([test_y.size])
    # for i, v in enumerate(test_y):
    #     ff[i] = abs((v - test_predictions[i])/v)
    # plt.plot(ff)

    plt.figure(5)
    plt.plot(train_errors)
    plt.plot(test_errors)
    plt.xscale('log')
    plt.legend(['Train', 'Test'])
    plt.title('Cost RMSE')

    # plt.figure(6)
    # plt.plot(train_accuracies)
    # plt.plot(test_accuracies)
    # plt.xscale('log')
    # plt.legend(['Train', 'Test'])
    # plt.title('R^2')


    plt.show()





