import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from numpy.linalg import inv

"""DATASET LOADING AND PREPROCESSING"""

df = pd.read_csv("insurance.txt")
df.insert(0,'x_0',1)
df = df.sample(frac=1)

def standardize(num,mu,sd):
  return (num-mu)/sd

def preprocessing_data(df):
  
  age_mean = df.mean()['age']
  age_sd =df.std()['age']
  
  bmi_mean = df.mean()['bmi']
  bmi_sd = df.std()['bmi']
  
  children_mean = df.mean()['children']
  children_sd = df.std()['children']
  
  charges_mean = df.mean()['charges']
  charges_sd = df.std()['charges']  
  
  df['age'] = df['age'].apply(lambda num : standardize(num,age_mean,age_sd)) 
  df['bmi'] = df['bmi'].apply(lambda num : standardize(num,bmi_mean,bmi_sd)) 
  df['children'] = df['children'].apply(lambda num : standardize(num,children_mean,children_sd)) 
  df['charges']  = df['charges'].apply(lambda num : standardize(num,charges_mean,charges_sd)) 
  
  return df

df = preprocessing_data(df)

"""FUNCTIONS"""

#Predicted Value
def predict(theta,x):
  return np.dot(x,theta)

#RMSE
def rmse(y_obs,y_pred):
  return np.sqrt(((y_obs - y_pred) ** 2).mean())

#Cost Function
def cost_fun(pred,y):
  return (np.sum((pred-y)**2))/2

#Normal Eqaution
def normal_eqn(train_X,train_Y):
  return inv(train_X.transpose().dot(train_X)).dot(train_X.transpose()).dot(train_Y)

#Gradient Descent
def grad_descent(x, y, alpha, epochs):
  feature_count = np.size(x[0])
  total_cases = np.size(x)
  theta = np.random.rand(4)

  pred= np.dot(x,theta)
  diff = np.subtract(pred,y)
  theta = theta - (1/total_cases)*alpha*(x.T.dot(diff))
  cost= 1/(2*total_cases) * np.sum(np.square(diff))
  
  for i in range(epochs):
    pred= np.dot(x,theta)
    diff = np.subtract(pred,y)
    theta = theta - (1/total_cases)*alpha*(x.T.dot(diff))
    cost_temp= 1/(2*total_cases) * np.sum(np.square(diff))

  return theta

#Stochastic Gradient Descent
def stoch_grad_des(x, y, alpha, epochs):
  theta = np.random.randn(4)
  l = len(y)
  for i in range(epochs) :
    rand_num = np.random.randint(0,l)
    pred= np.dot(x[rand_num],theta)
    diff = np.subtract(pred,y[rand_num])
    theta = theta - alpha*(x[rand_num].T.dot(diff))
    cost_temp= 1/(2*l) * np.sum(np.square(np.dot(x,theta)-y))
    
  return theta

"""TRAINING AND TESTING"""

training_errors_normal = []
training_errors_gd = []
training_errors_sgd = []
train_sse_normal = []
train_sse_gd = []
train_sse_sgd = []
train_sse_normal = []
test_sse_gd = []
test_sse_sgd = []
test_sse_normal = []
test_errors_gd = []
test_errors_sgd = []
test_errors_normal = []
parameters_list_normal = []
parameters_list_gd = []
parameters_list_sgd = []

for i in range(20):
  
  #Training 
  df = df.sample(frac=1, random_state=i*100)
  df = df.reset_index(drop=True)
  train_ratio = int(0.7*len(df))

  train_X = np.array(df.drop(['charges'], axis=1)[:train_ratio])
  test_X = np.array(df.drop(['charges'], axis=1)[train_ratio:])
  train_Y = np.array(df['charges'][:train_ratio])
  test_Y = np.array(df['charges'][train_ratio:])

  parameters = normal_eqn(train_X, train_Y)
  parameters_list_normal.append(parameters)
  
  parameters = grad_descent(train_X,train_Y, 0.001, 10000)
  parameters_list_gd.append(parameters)

  parameters = stoch_grad_des(train_X, train_Y, 0.001, 10000)
  parameters_list_sgd.append(parameters)

  y_train_predict = predict(parameters_list_normal[i],train_X)
  tr_error = rmse(train_Y, y_train_predict)
  training_errors_normal.append(tr_error)
  train_sse_i = cost_fun(y_train_predict,train_Y)
  train_sse_normal.append(train_sse_i)

  y_train_predict = predict(parameters_list_gd[i],train_X)
  tr_error = rmse(train_Y, y_train_predict)
  training_errors_gd.append(tr_error)
  train_sse_i = cost_fun(y_train_predict,train_Y)
  train_sse_gd.append(train_sse_i)

  y_train_predict = predict(parameters_list_sgd[i],train_X)
  tr_error = rmse(train_Y, y_train_predict)
  training_errors_sgd.append(tr_error)
  train_sse_i = cost_fun(y_train_predict,train_Y)
  train_sse_sgd.append(train_sse_i)


 
  #Testing
  y_test_predict = predict(parameters_list_normal[i],test_X)
  test_sse_i = cost_fun(y_test_predict,test_Y)
  test_sse_normal.append(test_sse_i)
  test_error = rmse(test_Y , y_test_predict)
  test_errors_normal.append(test_error)

  y_test_predict = predict(parameters_list_gd[i],test_X)
  test_sse_i = cost_fun(y_test_predict,test_Y)
  test_sse_gd.append(test_sse_i)
  test_error = rmse(test_Y , y_test_predict)
  test_errors_gd.append(test_error)

  y_test_predict = predict(parameters_list_sgd[i],test_X)
  test_sse_i = cost_fun(y_test_predict,test_Y)
  test_sse_sgd.append(test_sse_i)
  test_error = rmse(test_Y , y_test_predict)
  test_errors_sgd.append(test_error)

"""RESULTS"""

#Normal Equation
print('NORMAL EQUATION METHOD RESULTS:')
print("Training data :  SSE mean = {}  variance of SSE = {}".format(np.mean(train_sse_normal),np.var(train_sse_normal)))
print("Test data     :  SSE mean = {}  variance of SSE = {}".format(np.mean(test_sse_normal),np.var(test_sse_normal)))
print("Training data :  mean RMSE = {} variance RMSE = {}".format(np.mean(training_errors_normal),np.var(training_errors_normal)))
print("Test data     :  mean RMSE = {} variance RMSE = {}".format(np.mean(test_errors_normal),np.var(test_errors_normal)))
print('\n WEIGHTS 20 MODELS (Normal Equation)')
for index,i in enumerate(parameters_list_gd):
  print("Model {} :".format(index+1),i)

#Gradient Descent 
print('\nGRADIENT DESCENT METHOD RESULTS:')
print("Training data :  SSE mean = {}  variance of SSE = {}".format(np.mean(train_sse_gd),np.var(train_sse_gd)))
print("Test data     :  SSE mean = {}  variance of SSE = {}".format(np.mean(test_sse_gd),np.var(test_sse_gd)))
print("Training data :  mean RMSE = {} variance RMSE = {}".format(np.mean(training_errors_gd),np.var(training_errors_gd)))
print("Test data     :  mean RMSE = {} variance RMSE = {}".format(np.mean(test_errors_gd),np.var(test_errors_gd)))
print('\n WEIGHTS 20 MODELS (Gradient Descent)')
for index,i in enumerate(parameters_list_gd):
  print("Model {} :".format(index+1),i)

#Stochastic Gradient Descent 
print('\nSTOCHASTIC GRADIENT DESCENT METHOD RESULTS:')
print("Training data :  SSE mean = {}  variance of SSE = {}".format(np.mean(train_sse_sgd),np.var(train_sse_sgd)))
print("Test data     :  SSE mean = {}  variance of SSE = {}".format(np.mean(test_sse_sgd),np.var(test_sse_sgd)))
print("Training data :  mean RMSE = {} variance RMSE = {}".format(np.mean(training_errors_sgd),np.var(training_errors_sgd)))
print("Test data     :  mean RMSE = {} variance RMSE = {}".format(np.mean(test_errors_sgd),np.var(test_errors_sgd)))
print('\n WEIGHTS 20 MODELS (Stochastic Gradient Descent)')
for index,i in enumerate(parameters_list_sgd):
  print("Model {} :".format(index+1),i)

"""PLOTS"""

for k in range(3):
  df = df.sample(frac=1, random_state=k*100)
  df = df.reset_index(drop=True)
  train_ratio = int(0.7*len(df))
  train_X = np.array(df.drop(['charges'], axis=1)[:train_ratio])
  train_Y = np.array(df['charges'][:train_ratio])
  alpha = [1e-2, 1e-3, 1e-4]
  feature_count = np.size(train_X[0])
  total_cases = np.size(train_X)
  theta = np.random.rand(4)
  error_fun = []
  for i in range(10000):
    pred= np.dot(train_X,theta)
    diff = np.subtract(pred,train_Y)
    theta = theta - (1/total_cases)*alpha[k]*(train_X.T.dot(diff))
    cost_temp= 1/(2*total_cases) * np.sum(np.square(diff))
    error_fun.append(cost_temp)
  print('\n\nCOST FUNCTION VALUES: GRADIENT DESCENT LR=', alpha[k],'\n')
  for n in range(40):
    print("Cost after {} epochs :".format(n*250), error_fun[n*250])

  plt.plot(error_fun)
  plt.xlabel('EPOCHS', fontsize = 12) 
  plt.ylabel('ERROR', fontsize = 12)
  print('\n                   Lerning rate = ',alpha[k])
  plt.title('GRADIENT DESCENT', fontsize = 18)
  plt.show()

for k in range(3):
  df = df.sample(frac=1, random_state=k*100)
  df = df.reset_index(drop=True)
  train_ratio = int(0.7*len(df))
  train_X = np.array(df.drop(['charges'], axis=1)[:train_ratio])
  train_Y = np.array(df['charges'][:train_ratio])
  alpha = [0.01 , 0.001 , 0.0001 ]
  feature_count = np.size(train_X[0])
  total_cases = np.size(train_X)
  theta = np.random.randn(4)
  error_fun = []
  l = len(train_Y)
  
  for i in range(10000):
    rand_num = np.random.randint(0,l) 
    pred= np.dot(train_X[rand_num],theta)
    diff = np.subtract(pred,train_Y[rand_num])
    theta = theta - alpha[k]*(train_X[rand_num].T.dot(diff))
    cost_temp= 1/(2*l) * np.sum(np.square(np.dot(train_X,theta)-train_Y))
    error_fun.append(cost_temp)

  print('\n\nCOST FUNCTION VALUES: STOCHASTIC GRADIENT DESCENT LR=', alpha[k],'\n')
  for n in range(40):
    print("Cost after {} epochs :".format(n*250), error_fun[n*250])
  plt.plot(error_fun)
  plt.xlabel('EPOCHS', fontsize = 12) 
  plt.ylabel('ERROR', fontsize = 12)
  print('\n                   Learning rate = ',alpha[k])
  plt.title('STOCHASTIC GRADIENT DESCENT', fontsize = 18)
  plt.show()
