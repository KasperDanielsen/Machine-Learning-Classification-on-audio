import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt
class PlotCallback(keras.callbacks.Callback):
    
    def __init__(self,max_epochs,print_every=5):
        self.max_epochs = max_epochs
        self.print_every = print_every
        
    def plot(self, epoch):
        #score = model.evaluate(X_test, y_test, verbose=0)
        print("Epoch {} out of {}".format(epoch, self.max_epochs))
        print('Test loss: {:.2f}'.format(self.val_loss[-1]))
        print('Test accuracy: {:.2f}%'.format(self.val_acc[-1]*100))
        
        # Accuracy plot
        plt.plot(self.acc)
        plt.plot(self.val_acc)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        # Loss plot
        plt.plot(self.loss)
        plt.plot(self.val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        clear_output(wait=True)

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    # This function is called at the end of training
    def on_train_end(self, logs={}):
        self.plot(self.max_epochs)
        
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        if epoch % self.print_every == 0:
            self.plot(epoch)

def plot_confusion_matrix(cm, class_names=None):
    classes = len(cm)

    # Calculate accuracy for the plot title
    accuracy = 100*cm.diagonal().sum()/cm.sum()
    error_rate = 100-accuracy


    # Plot the confusion matrix
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='None')
    plt.colorbar(format='%.2f')
    plt.xticks(range(classes), labels=class_names)
    plt.yticks(range(classes), labels=class_names)

    plt.xlabel('Predicted class')
    plt.ylabel('Actual class')
    plt.title('Confusion matrix (Accuracy: {:.2f}%, Error Rate: {:.2f}%)'.format(accuracy, error_rate))
    plt.plot()

    # Fill the plot with the corresponding values
    threshold = cm.max() / 2.0 # Threshold for using white or black text
    for x in range(classes):
        for y in range(classes):
            plt.text(y, x, format(cm[x, y], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[x, y] > threshold else "black") # Some pretty printing to make it more legible
