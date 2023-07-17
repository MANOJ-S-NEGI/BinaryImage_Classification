# BinaryImage_Classification:
## Project: DOG_V/s_CAT
date: 23-JULY-2023


## DATA DESCRIPTION:
```
The Dogs vs. Cats dataset is a standard computer vision dataset that involves classifying photos as either containing a dog or a cat.
Download dataset: https://www.kaggle.com/c/dogs-vs-cats

the dataset contains 20000+ images of Dogs and cats
```

## Project Description:

```
1. Programming Language used: python
2. For Results web framework used:  FastAPI

the dataset contains 20000+ images of Dogs and cats for the training of the model.
the accuracy achieved 91% while the model saved on the best weights with a validation loss of 23% and validation accuracy of 89%

>> Run requirements.txt as:
 pip install -r requirements.txt
>> Run main.py then Copy the URL  http://127.0.0.1:8000/docs upload an image of a dog or a cat and check the predictions
```




```
IMP NOTE:  
If an openMP runtime error occurred, due to the same lib with more than one name using any one solution could solve the problem.

# SOL_1: fixed it by deleting the libiomp5md.dll duplicate file from Anaconda environment folder
OR
# SOL_2. Set env: <os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE">, Not Recommended
OR
#Sol_3: conda install nomkl --channel conda-forge

```



