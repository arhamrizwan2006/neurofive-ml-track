## Week 3 Task 1 – Model Evaluation & Tuning

### Why Accuracy Alone Can Be Misleading

About 59% of passengers did not survive, so a model that always guesses "did not
survive" would get ~59% accuracy without learning anything. Our model's 81% shows
real learning, but precision/recall/F1 matter because they reveal how well the
model performs on each class separately, not just overall.

### Before vs After Tuning

| Metric | Before Tuning | After Tuning |
|---|---|---|
| Accuracy | 0.81 | 0.78 |
| Precision (Survived) | 0.79 | 0.76 |
| Recall (Survived) | 0.74 | 0.69 |
| F1-score (Survived) | 0.76 | 0.72 |
| CV Mean Accuracy | 0.791 | 0.796 |

Best hyperparameters (GridSearchCV): **C = 1, solver = liblinear**

### Conclusion

The tuned model scored slightly lower on the test set, but higher on
cross-validation — a more reliable measure since it averages across 5 splits.
This suggests the test-set drop is mostly random variance from this particular
split, not evidence that tuning hurt the model.
