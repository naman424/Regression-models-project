#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#importing the data
data=pd.read_csv('datasets_88705_204267_Real estate.csv')
X=data.iloc[:,1:7].values
y=data.iloc[:,7:8].values


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
# decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()
regressor.fit(X_train,y_train)


#evaluating the selected model
# decision tree regression
from sklearn.metrics import r2_score
r2=r2_score(y_test,regressor.predict(X_test))

#for decision tree regression: r^2=0.3166

#visualising the test set results
xvals=np.arange(0,len(y_test))
plt.plot(xvals,y_test,color='blue')
plt.plot(xvals,regressor.predict(X_test),color='red')
plt.title('comparing the test set results with the predicted results')
plt.legend(['test set results','predicted set results'])
