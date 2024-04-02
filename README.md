# DCA
Code in Pytorch for our article Dual Classifier Adaptation: Source-Free UDA via Adaptive Pseudo-labels Learning

# Environment Requirement
- python 3.8.10
- pytorch 1.11.0
- torchvision 0.12.0
- cuda 10.1
- numpy, sklearn, scipy, tqdm

# Dataset
- Please manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)from the official websites, and modify the path of images in each '.txt' under the folder './data/'. [**How to generate such txt files could be found in https://github.com/tim-learn/Generate_list **]

# Training
Firstly, you need to train the source model by running this code
```
python train_source_31.py
```
Then, you can run the code of DCA to perform SFDA,
```
python train_target_31.py
```
