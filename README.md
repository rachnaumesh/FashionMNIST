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
- **Optimizer**: Adam (with or without weight decay depending on the model)
- **Dropout rate**: 0.5

We split the training data into a train-validation set using an 80-20 split and selected the best model based on validation accuracy.

## Convergence Graphs

The following graphs show the training and test accuracies for each technique and the baseline (no regularization) over the epochs.

![Convergence Graphs](convergence_graphs-2.png)

## Summary Table

| Model                                    | Final Train Accuracy (%) | Final Test Accuracy (%) |
|------------------------------------------|--------------------------|-------------------------|
| No Regularization                        | 95.70                    | 89.77                   |
| Dropout (0.5)                            | 90.78                    | 90.61                   |
| Weight Decay                             | 95.39                    | 90.31                   |
| Batch Normalization (before ReLU)        | 96.17                    | 90.23                   |

## Conclusions

1. **No Regularization**: Serves as a baseline for comparison. The model achieves high train accuracy but generalizes less well than other models.
2. **Dropout**: Helps prevent overfitting by reducing interdependent learning between neurons. The test accuracy is the highest among the regularization techniques.
3. **Weight Decay**: Prevents overfitting by adding a penalty term to the loss function, which discourages large weights. This technique strikes a balance between high train and test accuracy.
4. **Batch Normalization**: Improves training stability and speed by normalizing layer inputs. It achieves the highest train accuracy but slightly lower test accuracy compared to dropout.

## How to Train and Test

1. **Train the Model**: Use the command below for training different regularization techniques:
    ```bash
    python train.py --model_type [base|dropout|weight_decay|batch_norm]
    ```
    Example:
    ```bash
    python train.py --model_type dropout
    ```

2. **Test the Model**: After training, you can test the model using saved weights:
    ```bash
    python test.py --model_type [base|dropout|weight_decay|batch_norm] --weights_path ./saved_models/[model_name]_best.pth
    ```

## Hyperparameters

- **Batch size**: 64
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Dropout rate**: 0.5 for the dropout model
- **L2 Regularization (Weight Decay)**: 1e-4

## Reference

For the original LeNet-5 architecture, please refer to the paper by LeCun et al., 1998.
