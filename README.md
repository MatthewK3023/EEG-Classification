# EEG-Classification
In this implementation, I use Pytorch & NUMPY ot building a simple EEG classification models
which are EEGNet, DeepConvNet [1] with BCI competition dataset.

### Model Architecture
#### EEGNet
Overall visualization of the EEGNet architecture
<img src="./img/EEGNet_paper.png" width="90%">
<img src="./img/EEGNet_arch.png" width="90%">
Reference: Depthwise Separable Convolution
https://towardsdatascience.com/a-basic-introduction-to-separableconvolutionsb99ec3102728

#### DeepConvNet
Overall visualization of the EEGNet architecture
<img src="./img/DeepConvNet_paper.png" width="90%">
<img src="./img/DeepConvNet_arch.png" width="90%">

### Prepare Data
The training data and testing data have been preprocessed and named
[S4b_train.npz, X11b_train.npz] and [S4b_test.npz, X11b_test.npz]
respectively. Please download the preprocessed data and put it in the same
folder. To read the preprocessed data, refer to the “dataloader.py”.
<img src="./img/Dataset_example.png" width="90%">

### Result Comparison
#### EEGNet
<img src="./img/EEGNet_result.png" width="90%">

#### DeepConvNet
<img src="./img/DeepConvNet_result.png" width="90%">