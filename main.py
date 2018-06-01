import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
import datetime
import numpy as np
import fbprophet
from sklearn.metrics import mean_squared_error
import sys, os

def forecasting(file_name):
    # reading data
    data_set = pd.read_csv( file_name + "_ticker.csv",header=None)
    book_file = pd.read_csv( file_name + "_book.csv",header=None)
    trades_file = pd.read_csv( file_name + "_trades.csv",header=None)
    data_set['price'] = (data_set[7] + data_set[8]) / 2     # target
    data_set = data_set.drop([7 , 8], axis=1)

    # calculate important data from *_book and *_trades file
    order_volume = []
    for i in range(0,len(book_file)):
        sum_of_orders = 0
        for j in range(3,151,3):
            sum_of_orders = sum_of_orders + book_file.loc[i,j]
        order_volume.append(sum_of_orders)
    mapping = dict(enumerate(order_volume))
    data_set['order_volume'] = data_set[1].map(mapping)

    turnover = []
    for i in range(0,len(trades_file)):
        sum_of_turnover = 0
        for j in range(3,481,4):
            sum_of_turnover = sum_of_turnover + trades_file.loc[i,j]
        turnover.append(sum_of_turnover)
    mapping = dict(enumerate(turnover))
    data_set['turnover'] = data_set[1].map(mapping)
    data_set['transactionـprice'] = data_set[1].map(mapping)

    # print(data_set)



    # change date format
    date_list = []
    for i in range (0,len(data_set)):
        date_list.append(datetime.datetime.fromtimestamp(
             int(data_set[0][i])
             ).strftime('%Y-%m-%d %H:%M'))

    date_sries = pd.Series( date_list , name='date')
    data_set[0] = date_sries


    # preprocessing
    # set NaN value for noisy data in pandas
    data_set.loc[(data_set['price'] == 0) & (data_set[0] == -1), 'price'] = None
    # split for train and test
    test_set = data_set[round(len(data_set)*(0.9))+1:len(data_set)]
    data_set = data_set[0:round(len(data_set)*(0.9))]


    # use statistic model for forecasting
    data_set = data_set.rename(columns={ 0 : 'ds', 'price' : 'y'})
    A_prophet = fbprophet.Prophet()
    A_prophet.fit(data_set)

    A_forecast = A_prophet.make_future_dataframe(freq='H',periods=72)
    A_forecast = A_prophet.predict(A_forecast)

    A_prophet.plot(A_forecast, xlabel = 'Date', ylabel = 'Price')
    plt.title('prediction for ' + file_name);

    os.system('clear')

    # Evaluate prediction
    predicted_df = A_forecast.set_index('ds').join(test_set.set_index(0))
    predicted_df = predicted_df.dropna()
    predicted_df = predicted_df[["yhat","price"]]
    y_pred = predicted_df['yhat'].tolist()
    y_true = predicted_df['price'].tolist()
    print("======================================================")
    print("mean squar error for " + file_name + ":" , mean_squared_error(y_true, y_pred))
    print("======================================================\n")
    input("press any key to present forecasting plot...\nand to continue close the figure.")

    plt.plot(predicted_df.index , y_pred , color='red' , label="predicted value" )
    plt.plot(predicted_df.index , y_true , color='green' , label="true value" )
    plt.ylim((0, 150))
    plt.legend(loc="upper left", bbox_to_anchor=[0, 1],ncol=2, shadow=True, title="Legend", fancybox=True)
    plt.show()
    os.system('clear')



if __name__ == "__main__":
    letter = [ "A" ]
    for item in letter:
        forecasting(item)











#
