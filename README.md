# Training data

Date from 2017-08-27 to 2018-03-14, every minute's transaction count.

`200 days, every day has 1440 minutes, so totally there are 200*1440=288000 records`



<br>

# Predict

Predict next day every minute's`(1440 records)` transaction count.



<br>

# Use LSTM model

To see from here: <a href="http://qiaowei.tech/2017/12/20/%E5%9F%BA%E4%BA%8ELSTM%E8%BF%9B%E8%A1%8C%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B/" target="_blank">blog</a>



<br>

# Arch

```shell
save_data.py: get original data
tp_create_feature_vector_v1.py: transform original data to training data(features vector)
transaction_train_v3_k8s.py: training code used on k8s
transaction_train_v3.py: training code

predict: predicting code
```



