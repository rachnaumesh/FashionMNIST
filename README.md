# FashionMNIST

## Summary Table

| Model                                    | Final Train Accuracy (%) | Final Test Accuracy (%) |
|------------------------------------------|--------------------------|-------------------------|
| No Regularization                        | 95.70                    | 89.77                   |
| Dropout (0.5)                            | 90.78                    | 90.61                   |
| Weight Decay                             | 95.39                    | 90.31                   |
| Batch Normalization (before ReLU)        | 96.17                    | 90.23                   |

---

## Conclusions

1. **No Regularization**: Serves as a baseline for comparison.
2. **Dropout**: Helps prevent overfitting by reducing interdependent learning between neurons.
3. **Weight Decay**: Prevents overfitting by adding a penalty term to the loss function, discouraging large weights.
4. **Batch Normalization**: Improves training stability and speed by normalizing layer inputs.

---

## Comparison

- Dropout and Weight Decay typically show lower training accuracy but better generalization (higher test accuracy).
- Batch Normalization often leads to faster convergence and can improve both training and test accuracy.
- The effectiveness of each technique may vary depending on the specific dataset and model architecture.
- A combination of these techniques might yield the best results in practice.
