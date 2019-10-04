### How to run

0. Please create a folder named dataset in current path and put train.csv and test.csv into this folder

1. Run the below code under **src** directory and in order

The first time you should run **sh run.sh** to extract features. After that, you only need to run **make train-lgb-fs** to run model
```
sh run.sh
```
```

```
make train-lgb-fs
```
### Reuslt
By defualt, it will generate the below file in a folder called result

* submission.csv
* log
* feature_importance.csv
* lgbm_importances.png
