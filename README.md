```
IMP NOTE:  
If openMP runtime error occured, 

# SOL_1: fixed it by deleting the libiomp5md.dll duplicate file from Anaconda environment folder
OR
# SOL_2. Set env: <os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE">, Not Recommended
OR
#Sol_3: conda install nomkl --channel conda-forge

```



