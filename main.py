import pandas as pd
import numpy as np

# returns 2D array
# inner array = data1 = [1,feature1,feature2,...] 
# outer array = [data1,data2,data3,...]
def dataset_split_feature(start_index,end_index,data_file,num_feature):
  data = []
  for i in range(start_index,end_index):
    # [1] for bias term
    x = [1] + [feature for feature in data_file.iloc[i][:num_feature]]
    data.append(x)
  return np.array(data) 

# returns 2D array
# inner array = [data1's target1, data2's target1, data3's target1, ...]
# outer array = [all data's target1, all data's target2, ...]
def dataset_split_target(start_index,end_index,data_file,num_feature,num_target):
  all_target = []
  for i in range(num_feature,num_feature+num_target):
    target = []
    for data in range(start_index,end_index):
      target.append(data_file.iloc[data, i])
    all_target.append(target)
  return np.array(all_target)

def print_weights(weight_list,num_target):
  for i in range(num_target):
    print("Y"+str(i))
    for j in range(len(weight_list[i])): 
      print("X"+str(j)+":", weight_list[i][j]) # X0 is bias term
    print()

# NEXT 3 FUNCTIONS' PARAMETERS
# y_list = 1D array containing specific target e.g. array of target1 (Y1)
# w_list = 1D array of weights 
# x_list = 2D array with inner = [1,feature1,feature2,...] 
def sum_squared_error(y_list,w_list,x_list):
  sum = 0
  for data in range(len(x_list)):
    sum += (y_list[data] - ((w_list)@(x_list[data]))) ** 2
  return sum

def mean_absolute_error(y_list,w_list,x_list):
  sum = 0
  num = len(x_list)
  for data in range(num):
    sum += abs(y_list[data] - ((w_list)@(x_list[data])))
  return sum/num

def evaluate_loss(y_list,w_list,x_list):
  MSE, RMSE, MAE = [], [], []
  num_data = len(x_list)
  num_target = len(y_list)
  for i in range(num_target):
    MSE.append((1/num_data)*sum_squared_error(y_list[i],w_list[i],x_list))
  MSE = np.array(MSE)
  RMSE = np.sqrt(MSE)
  for j in range(num_target):
    MAE.append(mean_absolute_error(y_list[j],w_list[j],x_list))
  for k in range(num_target):
    print("Y"+str(k+1))
    print("Mean squared error:", MSE[k])
    print("Root mean squared error:", RMSE[k])
    print("Mean absolute error:", MAE[k],"\n")

#---------------------------------------------------------------------
df = pd.read_excel("ENB2012_data.xlsx")

# TRAINING:VALIDATION:TEST = 7:1:2 
num_data = len(df)
num_train = int(0.7 * num_data)
num_valid = int(0.1 * num_data)
num_test = num_data - num_train - num_valid

# COUNT THE NUMBER OF FEATURES AND TARGET
num_feature = 0
num_target = 0
for h in df.columns:
  if h[0] == 'X':
    num_feature += 1
  elif h[0] == 'Y':
    num_target += 1

# SPLITTING OF DATA
start = 0
end = num_train
train_A = dataset_split_feature(start,end,df,num_feature)
train_Y = dataset_split_target(start,end,df,num_feature,num_target)

start = end             # end of final data for training is the start of data for validating
end = start + num_valid # gets the wanted number of data wanted for this step (num_valid)
validate_A = dataset_split_feature(start,end,df,num_feature)
validate_Y = dataset_split_target(start,end,df,num_feature,num_target)

start = end
end = start + num_test
test_A = dataset_split_feature(start,end,df,num_feature)
test_Y = dataset_split_target(start,end,df,num_feature,num_target)
# END OF SPLITTING DATA

## LINEAR REGRESSION - TRAINING
weight_list = [] # weight_list will be a 2D array consisting of arrays for each target's set of weights
# FOR LOOP: calculate different weights per target
for target in range(num_target):
  weight = ((np.linalg.inv((train_A.T)@(train_A)))@(train_A.T))@train_Y[target]
  weight_list.append(weight)
weight_list = np.array(weight_list)

# LINEAR LOSS FUNCTION - VALIDATE
print("LINEAR: VALIDATE", 40*'-')
for i in range(num_target):
  print("Y"+str(i+1),end=' - ')
  print("Mean squared error: ",(1/num_valid)*sum_squared_error(validate_Y[i],weight_list[i],validate_A))

# MSE, RMSE, MAE - TEST
print("\nLINEAR: TEST", 44*'-')
evaluate_loss(test_Y,weight_list,test_A)
print_weights(weight_list,num_target)

# RIDGE REGRESSION - TRAINING
weight_list_dict = dict() # key: lambda/l2, value: 2D array consisting of arrays for each target's set of weights
l2_min = 0.0001           # minimum lambda chosen 
l2_max = 1
l2_step = l2_min
for l2 in np.arange(l2_min, l2_max+l2_step, l2_step):
  weight_list = []
  for target_num in range(num_target):
    weight = (np.linalg.inv(((train_A.T)@(train_A))+(l2*np.identity(num_feature+1)))@(train_A.T))@train_Y[target_num]
    weight_list.append(weight)
  weight_list_dict[l2] = np.array(weight_list)

# RIDGE LOSS FUNCTION - VALIDATE
print("RIDGE: VALIDATE", 40*'-')
best_l2 = l2_min
min_error = 1e300
for l2,weight_list in weight_list_dict.items():
  sum = 0 # sum will be the sum of each target's squared error
          # lambda or l2 will be chosen based on least total sum
  for j in range(num_target):
    sum += sum_squared_error(validate_Y[j],weight_list[j],validate_A) + l2*np.linalg.norm(weight_list[j])
  if sum < min_error:
    min_error = sum
    best_l2 = l2

print("Lambda =", best_l2)
print("L2 penalty =",best_l2*np.linalg.norm(weight_list_dict[best_l2][i]))
for i in range(num_target):
  print("Y"+str(i+1),end=' - ')
  print("Sum squared error + L2 penalty: ",sum_squared_error(validate_Y[i],weight_list_dict[best_l2][i],validate_A) + best_l2*np.linalg.norm(weight_list_dict[best_l2][i]))
# MSE, RMSE, MAE - TEST
print("\nRIDGE: TEST", 44*'-')
evaluate_loss(test_Y,weight_list_dict[best_l2],test_A)
print_weights(weight_list_dict[best_l2],num_target)





