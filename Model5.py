#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#importing the data
data=pd.read_csv('datasets_88705_204267_Real estate.csv')
X=data.iloc[:,1:7].values
y=data.iloc[:,7:8].values


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


#analysing the data
corr_matrix=data.corr()
from pandas.plotting import scatter_matrix
attributes=['X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude','Y house price of unit area']
scatter_matrix(data[attributes])
#from the scatter matrix, we can see that there doesn't exist a strong 
#correlation between any 2 variables



#selecting a suitable model for the data
# svr regression
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X_train,y_train)


#evaluating the selected model
# svr regression
from sklearn.metrics import r2_score
r2=r2_score(sc_y.inverse_transform(y_test),sc_y.inverse_transform(regressor.predict(X_test)))

#for svr regression: r^2=0.6626

#visualising the test set results
xvals=np.arange(0,len(y_test))
plt.plot(xvals,sc_y.inverse_transform(y_test),color='blue')
plt.plot(xvals,sc_y.inverse_transform(regressor.predict(X_test)),color='red')
plt.title('comparing the test set results with the predicted results')
plt.legend(['test set results','predicted set results'])
