## RBF Kernel

    $ python -m train -g -v -m
    INFO    Loading dataset from data.h5
    INFO    Performing grid search.
    Fitting 5 folds for each of 8 candidates, totalling 40 fits
    INFO    Search time: 850.77s
    INFO    Search Accuracy: 0.9447999999999999
    INFO    Best parameters: {
      "clf__C": 10,
      "clf__gamma": "auto",
      "clf__kernel": "rbf"
    }

    INFO    Training an SVM classifier.
    INFO    Training time: 291.04s
    INFO    Metrics computed on the test set
                  precision    recall  f1-score   support

               0       0.99      0.99      0.99       980
               1       0.99      0.99      0.99      1135
               2       0.96      0.97      0.96      1032
               3       0.97      0.97      0.97      1010
               4       0.98      0.97      0.97       982
               5       0.96      0.97      0.96       892
               6       0.98      0.98      0.98       958
               7       0.94      0.97      0.96      1028
               8       0.97      0.96      0.96       974
               9       0.97      0.96      0.96      1009

        accuracy                           0.97     10000
       macro avg       0.97      0.97      0.97     10000
    weighted avg       0.97      0.97      0.97     10000

    Confusion matrix:
    [[ 966    0    3    3    1    2    3    1    1    0]
     [   0 1127    3    0    0    1    2    1    1    0]
     [   5    3 1000    1    1    0    1   13    7    1]
     [   0    0    6  981    1    7    0    7    6    2]
     [   0    0    7    0  949    2    5    8    2    9]
     [   3    0    2    6    1  861    4    7    6    2]
     [   3    3    2    0    2    6  938    2    2    0]
     [   0    3   10    3    1    1    0 1000    0   10]
     [   2    1    4    6    5   10    1   10  932    3]
     [   1    4    4    8    7    4    0   14    2  965]]

## Polynomial kernel

    $ python -m train -g -v -m -k poly
    INFO    Loading dataset from data.h5
    INFO    Performing grid search.
    Fitting 5 folds for each of 24 candidates, totalling 120 fits
    INFO    Search time: 8732.52s
    INFO    Search Accuracy: 0.9561
    INFO    Best parameters: {
      "clf__C": 100,
      "clf__degree": 3,
      "clf__gamma": "auto",
      "clf__kernel": "poly"
    }

    INFO    Training an SVM classifier.
    INFO    Training time: 714.43s
    INFO    Metrics computed on the test set
                  precision    recall  f1-score   support

               0       0.98      0.99      0.99       980
               1       0.99      1.00      0.99      1135
               2       0.97      0.97      0.97      1032
               3       0.98      0.98      0.98      1010
               4       0.97      0.98      0.98       982
               5       0.97      0.97      0.97       892
               6       0.98      0.97      0.98       958
               7       0.98      0.97      0.97      1028
               8       0.95      0.97      0.96       974
               9       0.97      0.96      0.97      1009

        accuracy                           0.98     10000
       macro avg       0.98      0.98      0.98     10000
    weighted avg       0.98      0.98      0.98     10000

    Confusion matrix:
    [[ 971    0    1    1    0    1    3    1    2    0]
     [   0 1130    2    0    0    1    1    0    1    0]
     [   4    0 1002    1    2    0    2    7   14    0]
     [   0    0    2  990    1    3    2    4    7    1]
     [   1    0    2    0  964    0    4    1    2    8]
     [   2    1    3    7    1  868    3    0    5    2]
     [   3    3    1    1    5    7  933    0    5    0]
     [   1    4   10    1    5    1    0  995    2    9]
     [   2    0    4    5    4    4    0    2  948    5]
     [   3    3    1    5   10    6    0    4    8  969]]
