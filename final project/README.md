
## Final Project
* #### **題目**
> ###### 使用vix預測股價
###### -
* #### **組員**
> ###### 洪志穎、葉登元、徐上元
###### -
* #### **動機**
> ###### 股票的多空常被當作產業及公司表現的領先指標，當遇到股市大幅波動時，專業機構常都報導常提及VIX指數漲幅有多大、為多久來最高點等等，因此我們想透過機器學習了解VIX指數跟股票價格間是否有關連性及是否可從中套利與預測公司前景。
###### -
* #### **摘要**
> ###### 1.研究 random forest的參數設定及特性選擇歷史資訊充足、流動性高的個股
> ###### 2.選擇於市場上具規模、流動性高的個股，並使用台指vix指數進行預測
> ###### 3.使用幾種機器學習模型當作預測工具
> ###### 4.觀察標的股票預期漲跌與準確度
> ###### 5.修正與改善並設法提升準確性
　
###### -
* #### **開始預測**


載入模組

套件需求: pandas，sklearn中的RandomForest分類、回歸、交叉驗證。

```python

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd

```

資料來源為TEJ，期間為:2017/6/19~2018/6/15
將資料切為測試與訓練資料，測試資料為2017/6/19到2018/6/15。訓練資料為2018年6/15倒回去2017年6/19。
並修改訓練與測試資料欄位名稱為: VIX、VIX日期、VIX開盤價、VIX最高價、VIX最低價、VIX收盤價、台指期名稱、台指期日期、台指期報酬、台指期收盤、台指期成交量。

```python

test = pd.read_csv("test.csv", error_bad_lines=False)
train = pd.read_csv("train.csv", error_bad_lines=False)

train.columns=['vix','vix_date','vix_open','vix_high','vix_low','vix_close','future','fu_date','fu_ret','fu_colse','fu_vol']
test.columns=['vix','vix_date','vix_open','vix_high','vix_low','vix_close','future','fu_date','fu_ret','fu_colse','fu_vol']


```

我們願使用KD、MACD、RSI技術指標的值當作分類依據，因此需載numpy、talib，技術指標於talib運算後與之前欄位項目一同匯入t1，並移除空值與遺漏值。

```python

import talib
import numpy as np
def talib2df(talib_output):
    if type(talib_output) == list:
        ret = pd.DataFrame(talib_output).transpose()
    else:
        ret = pd.Series(talib_output)
    ret.index = test['vix_close'].index
    return ret;

t1 = {
    'close':test.vix_close.dropna().astype(float),
    'open':test.vix_open.dropna().astype(float),
    'high':test.vix_high.dropna().astype(float),
    'low':test.vix_low.dropna().astype(float),
    'volume': test.fu_volum.dropna().astype(float)    
}

KD = talib2df(talib.abstract.STOCH(t1, fastk_period=9))
MACD = talib2df(talib.abstract.MACD(t1))
RSI = talib2df(talib.abstract.RSI(t1))
t1=pd.DataFrame(t1)
t1 = pd.concat([test,KD,MACD,RSI], axis=1)
t1.columns=['vix','vix_date','vix_open','vix_high','vix_low','vix_close','future','fu_date','fu_ret','fu_colse','fu_vol','k','d','dif12','dif26','macd','rsi']

t1 = t1.query('vix_date > 20170803')
df=t1

```

先載入sklearn的模型選擇、預先處理、指標與ensemble模型後我們開始進行分類。我們將(交易量增加的)、(報酬率>0)、(波動率指數增加的)設為1; 因為想要用類別(不是連續)的變數，因此需要載入虛擬變數(dummy variables)，最後測試並建立volume的數據。

from sklearn import model_selection, ensemble, preprocessing, metrics

df['pre_vol']=(df.fu_vol - df.fu_vol.shift(1)) > 0
df['pre_ret']=(df.fu_ret -  0 ) > 0
df['pre_vix_close']=(df.vix_close - df.vix_close.shift(1)) > 0
df=df.dropna()


label_encoder = preprocessing.LabelEncoder()
encoded_label = label_encoder.fit_transform(df["future"])
encoded_label2 = label_encoder.fit_transform(df["vix"])
df['future']= encoded_label
df['vix'] = encoded_label2

selected_features = ['vix_date','vix_open','vix_high','vix_low','vix_close','fu_vol','k','d','macd','rsi','fu_ret','future']
pre_vol_X = df[selected_features]
pre_vol_y = df['pre_vol']
train_X, test_X, train_y, test_y = model_selection.train_test_split(pre_vol_X, pre_vol_y, test_size = 0.3)

接下來我們載入 random forest 模型並進行預測，我們發顯此模型預測的準確率為0.654，auc值(Area Under Curve)為0.656(auc越接近1越好，可顯示此分類算法的優劣程度)。

forest = ensemble.RandomForestClassifier(n_estimators = 300)
forest_fit = forest.fit(train_X, train_y)

test_y_predictedtest_y_p  = forest.predict(test_X)

accuracy = metrics.accuracy_score(test_y, test_y_predicted)
fpr, tpr, thresholds = metrics.roc_curve(test_y, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print('準確率: {}'.format(auc))
print('AUC值: {}'.format(accuracy))

用上述分類來預測台指期的交易量增加or減少，及預測成功機率。
最後再進行台指期報酬率的預測(賺錢or賠錢)，並顯示準確度與AUC值。

today_X = df[selected_features]
today_y_predicted = forest.predict(today_X)
proba = forest.predict_proba(today_X)


printprint(('隔日交易量: ''隔日交易量:   + format(np.where(today_y_predicted==True,'增','減')[0]))
print( '明日增加的機率: {}'.format(proba[0][1]))

selected_features_ret = ['vix_date','fu_date','vix_high','vix_low','fu_vol','k','d','macd','rsi','future']
pre_ret_X = df[selected_features_ret]
pre_ret_y = df['pre_ret']
train_X2, test_X2, train_y2, test_y2 = model_selection.train_test_split(pre_ret_X, pre_ret_y, test_size = 0.3)
forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(train_X2, train_y2)
test_y2_predicted = forest.predict(test_X2)
accuracy = metrics.accuracy_score(test_y2, test_y2_predicted)
fpr, tpr, thresholds = metrics.roc_curve(test_y2, test_y2_predicted)
auc = metrics.auc(fpr, tpr)
print('準確率: {}'.format(auc))
print('AUC值: {}'.format(accuracy))
today_X2 = df[selected_features_ret]
today_y2_predicted = forest.predict(today_X2)
proba = forest.predict_proba(today_X2)
print('預期隔日報酬: ' + format(np.where(today_y_predicted==True,'正','負')[0]))
print( '明日賺的機率: {}'.format(proba[0][1]))


