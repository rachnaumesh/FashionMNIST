# LeNet-5 on FashionMNIST

This project implements and trains the LeNet-5 network on the FashionMNIST dataset using PyTorch. The following regularization techniques are compared:

- **Dropout** (at the hidden layer)
- **Weight Decay (L2 Loss)**
- **Batch Normalization**

## Dataset

The FashionMNIST dataset can be downloaded from [this link](https://github.com/zalandoresearch/fashion-mnist). It consists of 60,000 training images and 10,000 test images. The dataset contains 28x28 grayscale images from 10 different classes (e.g., T-shirts, shoes).

## LeNet-5 Architecture

The LeNet-5 architecture follows the design from LeCun et al., 1998, with modifications for the FashionMNIST input dimensions and tasks. We have also made minor changes such as using ReLU as the activation function instead of the original Tanh.

### Model Variants

- **No Regularization**: Standard LeNet-5 model with no regularization techniques.
- **Dropout**: Applies dropout with a rate of 0.5 at the hidden layers.
- **Weight Decay**: Adds L2 regularization to the loss function with a penalty term.
- **Batch Normalization**: Applies batch normalization before the ReLU activations.

## Training Process

The models are trained for 20 epochs, and the following hyperparameters are used:

- **Batch size**: 64
- **Learning rate**: 0.001
- **Optimizer**: Adam 
- **Dropout rate**: 0.5

We split the training data into a train-validation set using an 80-20 split and selected the best model based on validation accuracy.

## Convergence Graphs

The following graphs show the training and test accuracies for each technique and the baseline (no regularization) over the epochs.

![Convergence Graphs](results/convergence_graphs-2.png)

## Summary Table

| Model                                    | Train Accuracy (%)       | Best Test Accuracy (%)  | Best Epoch              |
|------------------------------------------|--------------------------|-------------------------|-------------------------|
| No Regularization                        | 95.65                    | 90.69                   | 12                      |
| Dropout (0.5)                            | 90.52                    | 90.24                   | 19                      |
| Weight Decay                             | 95.27                    | 90.79                   | 17                      |
| Batch Normalization (before ReLU)        | 96.15                    | 90.88                   | 14                      |

## Conclusions

1. **Base Model**: Serves as a baseline for comparison. The model achieves high train accuracy but generalizes less well than other models.
2. **Dropout**: Helps prevent overfitting by reducing interdependent learning between neurons. 
3. **Weight Decay**: Prevents overfitting by adding a penalty term to the loss function, which discourages large weights. This technique strikes a balance between high train and test accuracy.
4. **Batch Normalization**: Improves training stability and speed by normalizing layer inputs. It achieves the highest train and test accuracies.

## How to Train and Test

1. **Training and Evaluating the Model**: 
To train and evaluate the models, the final block of code should be executed within the provided notebook. This block initializes four different model configurations and runs a training and testing loop for each one, over a predefined number of epochs.
```bash
models = {
    'Base Model': LeNet5().to(device),
    'Dropout': LeNet5Regularized(dropout_rate=dropout_rate).to(device),
    'Weight Decay': LeNet5().to(device),
    'Batch Normalization': LeNet5Regularized(dropout_rate=0, use_bn=True).to(device)
}
```
Once this block is run, each model will undergo training for the specified number of epochs. During each epoch, the code evaluates the model on both the training and test datasets to monitor performance and generalization. The best model for each configuration is saved based on test accuracy, and the results are used to generate convergence graphs and a final accuracy comparison table.

## Reference

For the original LeNet-5 architecture, please refer to the paper by LeCun et al., 1998.
