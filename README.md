# Stock-Price-Prediction-using-Simple-Neural-Network
This project aims to predict stock prices using machine learning techniques.

## Installation

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.

```python
import yfinance as yf
```

```python
from datetime import datetime
end = datetime.now()
start = datetime(end.year-10 , end.month , end.day)
```

```python
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(adj_close_price)
scaled_data
```

```python
import numpy as np
```

```python
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

import numpy as np
import pandas as pd
x_data, y_data = np.array(x_data), np.array(y_data)
```

```python
from keras.models import Sequential
from keras.layers import Dense, LSTM
```


## Usage

Run the Jupyter notebook to train the model and make predictions.


## Notebook Content Summary

Installing req library to get stock data

```python
pip install yfinance
```
Data Importing , Cleaning , Analysis

```python
import yfinance as yf
```
```python
from datetime import datetime
end = datetime.now()
start = datetime(end.year-10 , end.month , end.day)
```
```python
stock = "RELIANCE.NS"
rel_data = yf.download(stock ,start ,end)
```
```python
rel_data.head()
```
{'text/plain': '                  Open        High         Low       Close   Adj Close  \\\nDate                                                                     \n2014-06-13  503.020142  506.083160  490.402344  494.608276  466.453979   \n2014-06-16  495.568329  495.568329  483.293427  487.087891  459.361725   \n2014-06-17  483.727722  499.819977  483.727722  498.334198  469.967773   \n2014-06-18  497.397003  503.637329  484.893494  487.682220  459.922150   \n2014-06-19  490.996674  491.430969  471.612823  476.413055  449.294464   \n\n              Volume  \nDate                  \n2014-06-13   7323452  \n2014-06-16   7185688  \n2014-06-17   7912480  \n2014-06-18   9606352  \n2014-06-19  11563143  ', 

```python

rel_data.shape
```
{'text/plain': '(2463, 6)'}

```python
rel_data.describe()
```


```python
rel_data.info()
```
Graph Plotting for each Column VS Year

```python
import matplotlib.pyplot as plt
%matplotlib inline
```
```python
def plot_data(figsize, values , column_name):
    plt.figure()
    values.plot(figsize = figsize)
    plt.xlabel("year")
    plt.ylabel(column_name)
    plt.title(f"{column_name} of reliance stock ")
```
```python
rel_data.columns
```
{'text/plain': "Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')"}

```python
for column in rel_data.columns:
    plot_data((15,5),rel_data[column] ,column)
```










```python
for i in range(2014,2025):
    print(i,list(rel_data.index.year).count(i))
```
```python
rel_data['ma_for_250_days'] = rel_data['Adj Close'].rolling(250).mean()
```
```python
rel_data['ma_for_250_days']
```
{'text/plain': 'Date\n2014-06-13            NaN\n2014-06-16            NaN\n2014-06-17            NaN\n2014-06-18            NaN\n2014-06-19            NaN\n                 ...     \n2024-06-06    2603.679640\n2024-06-07    2606.352471\n2024-06-10    2609.061367\n2024-06-11    2611.682080\n2024-06-12    2614.274870\nName: ma_for_250_days, Length: 2463, dtype: float64'}

```python
rel_data['ma_for_250_days'][0:250].tail()
```
{'text/plain': 'Date\n2015-06-17           NaN\n2015-06-18           NaN\n2015-06-19           NaN\n2015-06-22           NaN\n2015-06-23    403.474999\nName: ma_for_250_days, dtype: float64'}

```python
plot_data((15,5), rel_data['ma_for_250_days'] , 'ma_for_250_days')
```


```python
rel_data['ma_for_100_days'] = rel_data['Adj Close'].rolling(100).mean()
```
```python
plot_data((15,5), rel_data[['Adj Close','ma_for_100_days']] , 'ma_for_100_days')
```

```python
plot_data((15,5), rel_data[['Adj Close','ma_for_100_days','ma_for_250_days']] , 'MA')
```


```python
rel_data['percentage_change'] = rel_data['Adj Close'].pct_change()
```
```python
rel_data[['Adj Close','percentage_change']].head()
```
{'text/plain': '             Adj Close  percentage_change\nDate                                     \n2014-06-13  466.453979                NaN\n2014-06-16  459.361725          -0.015205\n2014-06-17  469.967773           0.023089\n2014-06-18  459.922150          -0.021375\n2014-06-19  449.294464          -0.023108', 'text/html': '\n  <div id="df-b9b07952-40e4-46ae-b399-112e1b6b0e6a" class="colab-df-container">\n    <div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Adj Close</th>\n      <th>percentage_change</th>\n    </tr>\n    <tr>\n      <th>Date</th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>2014-06-13</th>\n      <td>466.453979</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>2014-06-16</th>\n      <td>459.361725</td>\n      <td>-0.015205</td>\n    </tr>\n    <tr>\n      <th>2014-06-17</th>\n      <td>469.967773</td>\n      <td>0.023089</td>\n    </tr>\n    <tr>\n      <th>2014-06-18</th>\n      <td>459.922150</td>\n      <td>-0.021375</td>\n    </tr>\n    <tr>\n      <th>2014-06-19</th>\n      <td>449.294464</td>\n      <td>-0.023108</td>\n    </tr>\n  </tbody>\n</table>\n</div>\n    <div class="colab-df-buttons">\n\n  <div class="colab-df-container">\n    <button class="colab-df-convert" onclick="convertToInteractive(\'df-b9b07952-40e4-46ae-b399-112e1b6b0e6a\')"\n            title="Convert this dataframe to an interactive table."\n            style="display:none;">\n\n  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">\n    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>\n  </svg>\n    </button>\n\n  <style>\n    .colab-df-container {\n      display:flex;\n      gap: 12px;\n    }\n\n    .colab-df-convert {\n      background-color: #E8F0FE;\n      border: none;\n      border-radius: 50%;\n      cursor: pointer;\n      display: none;\n      fill: #1967D2;\n      height: 32px;\n      padding: 0 0 0 0;\n      width: 32px;\n    }\n\n    .colab-df-convert:hover {\n      background-color: #E2EBFA;\n      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n      fill: #174EA6;\n    }\n\n    .colab-df-buttons div {\n      margin-bottom: 4px;\n    }\n\n    [theme=dark] .colab-df-convert {\n      background-color: #3B4455;\n      fill: #D2E3FC;\n    }\n\n    [theme=dark] .colab-df-convert:hover {\n      background-color: #434B5C;\n      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n      fill: #FFFFFF;\n    }\n  </style>\n\n    <script>\n      const buttonEl =\n        document.querySelector(\'#df-b9b07952-40e4-46ae-b399-112e1b6b0e6a button.colab-df-convert\');\n      buttonEl.style.display =\n        google.colab.kernel.accessAllowed ? \'block\' : \'none\';\n\n      async function convertToInteractive(key) {\n        const element = document.querySelector(\'#df-b9b07952-40e4-46ae-b399-112e1b6b0e6a\');\n        const dataTable =\n          await google.colab.kernel.invokeFunction(\'convertToInteractive\',\n                                                    [key], {});\n        if (!dataTable) return;\n\n        const docLinkHtml = \'Like what you see? Visit the \' +\n          \'<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>\'\n          + \' to learn more about interactive tables.\';\n        element.innerHTML = \'\';\n        dataTable[\'output_type\'] = \'display_data\';\n        await google.colab.output.renderOutput(dataTable, element);\n        const docLink = document.createElement(\'div\');\n        docLink.innerHTML = docLinkHtml;\n        element.appendChild(docLink);\n      }\n    </script>\n  </div>\n\n\n<div id="df-35799f96-658c-4fc2-a2d1-ff432ccf7b57">\n  <button class="colab-df-quickchart" onclick="quickchart(\'df-35799f96-658c-4fc2-a2d1-ff432ccf7b57\')"\n            title="Suggest charts"\n            style="display:none;">\n\n<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"\n     width="24px">\n    <g>\n        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>\n    </g>\n</svg>\n  </button>\n\n<style>\n  .colab-df-quickchart {\n      --bg-color: #E8F0FE;\n      --fill-color: #1967D2;\n      --hover-bg-color: #E2EBFA;\n      --hover-fill-color: #174EA6;\n      --disabled-fill-color: #AAA;\n      --disabled-bg-color: #DDD;\n  }\n\n  [theme=dark] .colab-df-quickchart {\n      --bg-color: #3B4455;\n      --fill-color: #D2E3FC;\n      --hover-bg-color: #434B5C;\n      --hover-fill-color: #FFFFFF;\n      --disabled-bg-color: #3B4455;\n      --disabled-fill-color: #666;\n  }\n\n  .colab-df-quickchart {\n    background-color: var(--bg-color);\n    border: none;\n    border-radius: 50%;\n    cursor: pointer;\n    display: none;\n    fill: var(--fill-color);\n    height: 32px;\n    padding: 0;\n    width: 32px;\n  }\n\n  .colab-df-quickchart:hover {\n    background-color: var(--hover-bg-color);\n    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n    fill: var(--button-hover-fill-color);\n  }\n\n  .colab-df-quickchart-complete:disabled,\n  .colab-df-quickchart-complete:disabled:hover {\n    background-color: var(--disabled-bg-color);\n    fill: var(--disabled-fill-color);\n    box-shadow: none;\n  }\n\n  .colab-df-spinner {\n    border: 2px solid var(--fill-color);\n    border-color: transparent;\n    border-bottom-color: var(--fill-color);\n    animation:\n      spin 1s steps(1) infinite;\n  }\n\n  @keyframes spin {\n    0% {\n      border-color: transparent;\n      border-bottom-color: var(--fill-color);\n      border-left-color: var(--fill-color);\n    }\n    20% {\n      border-color: transparent;\n      border-left-color: var(--fill-color);\n      border-top-color: var(--fill-color);\n    }\n    30% {\n      border-color: transparent;\n      border-left-color: var(--fill-color);\n      border-top-color: var(--fill-color);\n      border-right-color: var(--fill-color);\n    }\n    40% {\n      border-color: transparent;\n      border-right-color: var(--fill-color);\n      border-top-color: var(--fill-color);\n    }\n    60% {\n      border-color: transparent;\n      border-right-color: var(--fill-color);\n    }\n    80% {\n      border-color: transparent;\n      border-right-color: var(--fill-color);\n      border-bottom-color: var(--fill-color);\n    }\n    90% {\n      border-color: transparent;\n      border-bottom-color: var(--fill-color);\n    }\n  }\n</style>\n\n  <script>\n    async function quickchart(key) {\n      const quickchartButtonEl =\n        document.querySelector(\'#\' + key + \' button\');\n      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n      quickchartButtonEl.classList.add(\'colab-df-spinner\');\n      try {\n        const charts = await google.colab.kernel.invokeFunction(\n            \'suggestCharts\', [key], {});\n      } catch (error) {\n        console.error(\'Error during call to suggestCharts:\', error);\n      }\n      quickchartButtonEl.classList.remove(\'colab-df-spinner\');\n      quickchartButtonEl.classList.add(\'colab-df-quickchart-complete\');\n    }\n    (() => {\n      let quickchartButtonEl =\n        document.querySelector(\'#df-35799f96-658c-4fc2-a2d1-ff432ccf7b57 button\');\n      quickchartButtonEl.style.display =\n        google.colab.kernel.accessAllowed ? \'block\' : \'none\';\n    })();\n  </script>\n</div>\n\n    </div>\n  </div>\n', 'application/vnd.google.colaboratory.intrinsic+json': {'type': 'dataframe', 'summary': '{\n  "name": "rel_data[[\'Adj Close\',\'percentage_change\']]",\n  "rows": 5,\n  "fields": [\n    {\n      "column": "Date",\n      "properties": {\n        "dtype": "date",\n        "min": "2014-06-13 00:00:00",\n        "max": "2014-06-19 00:00:00",\n        "num_unique_values": 5,\n        "samples": [\n          "2014-06-16 00:00:00",\n          "2014-06-19 00:00:00",\n          "2014-06-17 00:00:00"\n        ],\n        "semantic_type": "",\n        "description": ""\n      }\n    },\n    {\n      "column": "Adj Close",\n      "properties": {\n        "dtype": "number",\n        "std": 7.921996749269424,\n        "min": 449.2944641113281,\n        "max": 469.9677734375,\n        "num_unique_values": 5,\n        "samples": [\n          459.3617248535156,\n          449.2944641113281,\n          469.9677734375\n        ],\n        "semantic_type": "",\n        "description": ""\n      }\n    },\n    {\n      "column": "percentage_change",\n      "properties": {\n        "dtype": "number",\n        "std": 0.021758198818590823,\n        "min": -0.023107574955398613,\n        "max": 0.023088664140153314,\n        "num_unique_values": 4,\n        "samples": [\n          0.023088664140153314,\n          -0.023107574955398613,\n          -0.015204618141307247\n        ],\n        "semantic_type": "",\n        "description": ""\n      }\n    }\n  ]\n}'}}

```python
plot_data((15,5), rel_data['percentage_change'] , 'percentage_change')
```

Data Processing

```python
adj_close_price = rel_data[['Adj Close']]
```
```python
max(adj_close_price.values),min(adj_close_price.values)
```
{'text/plain': '(array([3020.64990234]), array([349.54946899]))'}

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(adj_close_price)
scaled_data
```
{'text/plain': 'array([[0.04376642],\n       [0.04111124],\n       [0.04508191],\n       ...,\n       [0.97085476],\n       [0.95982936],\n       [0.96480851]])'}

```python
len(scaled_data)
```
{'text/plain': '2463'}

```python
import numpy as np
```
```python
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

import numpy as np
import pandas as pd
x_data, y_data = np.array(x_data), np.array(y_data)
```
```python
x_data[0] ,y_data[0]
```
{'text/plain': '(array([[0.04376642],\n        [0.04111124],\n        [0.04508191],\n        [0.04132105],\n        [0.03734229],\n        [0.0364384 ],\n        [0.03667242],\n        [0.04020735],\n        [0.03877884],\n        [0.03246769],\n        [0.03249999],\n        [0.03303265],\n        [0.0318382 ],\n        [0.03349265],\n        [0.03159608],\n        [0.03570397],\n        [0.03493727],\n        [0.02974793],\n        [0.03070026],\n        [0.03009499],\n        [0.02513968],\n        [0.02437297],\n        [0.0260597 ],\n        [0.02888439],\n        [0.02772223],\n        [0.02679411],\n        [0.03016759],\n        [0.03554255],\n        [0.03591381],\n        [0.03717281],\n        [0.03407373],\n        [0.03145083],\n        [0.03153958],\n        [0.03158801],\n        [0.02672149],\n        [0.02812576],\n        [0.02794014],\n        [0.0286342 ],\n        [0.02881983],\n        [0.02739941],\n        [0.02739941],\n        [0.02802892],\n        [0.02895702],\n        [0.03102307],\n        [0.0326533 ],\n        [0.03204804],\n        [0.03028058],\n        [0.029756  ],\n        [0.0302725 ],\n        [0.02986898],\n        [0.02947353],\n        [0.02953001],\n        [0.03033709],\n        [0.03282279],\n        [0.03438846],\n        [0.03559097],\n        [0.03472744],\n        [0.03468708],\n        [0.03663209],\n        [0.03587346],\n        [0.03310527],\n        [0.03367828],\n        [0.03306489],\n        [0.03099078],\n        [0.02722186],\n        [0.0287149 ],\n        [0.0313378 ],\n        [0.02981248],\n        [0.02918298],\n        [0.02503476],\n        [0.02490561],\n        [0.01922397],\n        [0.01977276],\n        [0.01975663],\n        [0.02178233],\n        [0.01879622],\n        [0.01862677],\n        [0.02032962],\n        [0.02322697],\n        [0.02419542],\n        [0.02375153],\n        [0.02429224],\n        [0.02429224],\n        [0.01940962],\n        [0.02052332],\n        [0.0200633 ],\n        [0.01914327],\n        [0.02177426],\n        [0.02108018],\n        [0.0199019 ],\n        [0.02273465],\n        [0.02719765],\n        [0.03063571],\n        [0.03092623],\n        [0.02914262],\n        [0.02739941],\n        [0.02559969],\n        [0.02694747],\n        [0.02635831],\n        [0.02558353]]),\n array([0.02556739]))'}

Splitting Data

```python
int(len(x_data)*0.7)
```
{'text/plain': '1654'}

```python
2462-100-int(len(x_data)*0.7)
```
{'text/plain': '708'}

```python
splitting_len = int(len(x_data)*0.7)
x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]

x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]
```
```python
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
```
```python
from keras.models import Sequential
from keras.layers import Dense, LSTM
```
```python
sq = Sequential()
sq.add(LSTM(128 , return_sequences=True , input_shape=(x_train.shape[1],1)))
sq.add(LSTM(64 , return_sequences= False))
sq.add(Dense(25))
sq.add(Dense(1))
```
```python
sq.compile(optimizer='adam',loss='mean_squared_error')
```
```python
sq.fit(x_train, y_train, batch_size=1, epochs = 2)
```
{'text/plain': '<keras.src.callbacks.History at 0x7e6580526f50>'}

```python
sq.summary()
```
```python
x_pred = sq.predict(x_test)
```
```python
x_pred
```


```python
inv_prediction = scaler.inverse_transform(x_pred)
inv_prediction
```

```python
inv_y_test_prediction = scaler.inverse_transform(y_test)
inv_y_test_prediction
```

```python
rmse= np.sqrt(np.mean(( inv_prediction - inv_y_test_prediction)**2))
rmse
```
{'text/plain': '46.2569794181678'}

```python
ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test_prediction.reshape(-1),
    'predictions': inv_prediction.reshape(-1)
 } ,
    index = rel_data.index[splitting_len+100:]
)
ploting_data.head()
```


```python
plot_data((15,5), ploting_data, 'test data')
```

```python
plot_data((15,5), pd.concat([adj_close_price[:splitting_len+100],ploting_data], axis=0), 'whole data')
```


```python
sq.save("Latest_stock_price_model.keras")
```
```python

```
