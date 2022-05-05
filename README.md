# MTAN
Pytorch Implementation for Multi-view Transfer Attention Network for Dementia Status Prediction

Results:(Training on ADNI-1 and testing on ADNI-2)
|  Methods | CDRSB |       | ADAS11 |       | ADAS13 |        |  MMSE |       |
|:--------:|:-----:|:-----:|:------:|:-----:|:------:|:------:|:-----:|:-----:|
|          |   CC  |  RMSE |   CC   |  RMSE |   CC   |  RMSE  |   CC  |  RMSE |
| ROI      | 0.303 | 2.343 |  0.374 | 8.136 |  0.397 | 11.616 | 0.361 | 3.111 |
| VBM      | 0.567 | 2.014 |  0.523 | 7.609 |  0.531 | 10.875 | 0.469 | 2.970 |
| VIT-mean | 0.537 | 1.944 |  0.562 | 8.027 |  0.591 | 10.831 | 0.516 | 2.831 |
| VIT-cls  | 0.568 | 1.885 |  0.564 | 7.116 |  0.584 |  9.718 | 0.516 | 2.603 |
| CNN      | 0.539 | 1.671 |  0.584 | 5.921 |  0.600 |  8.252 | 0.512 | 2.521 |
| wiseDNN  | 0.532 | 1.664 |  0.561 | 6.234 |  0.586 |  8.536 | 0.502 | 2.435 |
| MTAN     | 0.583 | 1.586 |  0.602 | 5.509 |  0.636 |  7.812 | 0.580 | 2.353 |

