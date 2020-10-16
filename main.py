import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def welcome():
    print("\n                      Welcome to Salary Prediction Sysytem")
    print("\nPress ENTER Key to Proceed")
    input()

def checkcsv():
    csv_files = []
    cur_dir = os.getcwd()
    content_list = os.listdir(cur_dir)
    for x in content_list:
        if x.split(".")[-1] == "csv":
            csv_files.append(x)
    if len(csv_files) == 0:
        return "No csv File Exist in The Directory"
    else:
        return csv_files
def display_and_select_csv(csv_files):
    i = 0
    for file_name in csv_files:
        print(i,"..",file_name)
        i += 1
    return csv_files[int(input("\nSelect The File to Create Machine Learning Maodel"))]

def graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred):
    plt.scatter(X_train,Y_train,color = "red",label = "Training Data")
    plt.plot(X_train,regressionObject.predict(X_train),color = "blue",label = "Best Fit")
    plt.scatter(X_test,Y_test,color = "green",label = "Test Data")
    plt.scatter(X_test,Y_pred,color = "black",label = "Predicted Data")
    plt.title("Salary VS Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.show()
    

def main():
    welcome()
    try:
       csv_files = checkcsv()
       if csv_files == "No csv File Exist in The Directory":
           raise FileNotFoundError()
       csv_file = display_and_select_csv(csv_files)
       print(csv_file,"is Selected")
       print("Reading csv File")
       print("\nCreating Data-Set")
       dataset = pd.read_csv(csv_file)
       print("Data-Set Created")
       X = dataset.iloc[:,:-1].values
       Y = dataset.iloc[:,-1].values
       s = float(input("\nEnter Test Data Size (Between 0 and 1)"))  ## 0.1 means 10%
       
       ## Creating Model
       
       X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = s)
       print("Model Creation is in Progress")
       regressionObject = LinearRegression()
       regressionObject.fit(X_train,Y_train)
       print("Model is Created")
       print("\nPress Enter Key to Predict Test Data in Trained Model")
       input()

       ## Testing The Model
       Y_predict = regressionObject.predict(X_test)
       i = 0
       print(X_test,"....",Y_test,"....",Y_predict)
       while i < len(X_test):
           print(X_test[i],"....",Y_test[i],"....",Y_predict)
           i += 1
       print("\nPress Enter Key to See Above Result in Graphical Format")
       input()
       graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_predict)

       r2 = r2_score(Y_test,Y_predict)
       print("\n\n                      Our model is %2.2f%% accurate" %(r2*100))
       print("\nNow You Can Predict Salaries of Employees Using Our Model")
       print("\nEnter Experiences of the Candidates in Years Separated By ,")

       exp = [float(e) for e in input().split(",")]
       ex = []
       for x in exp:
           ex.append([x])
       experience = np.array(ex)
       salaries = regressionObject.predict(experience)

       plt.scatter(experience,salaries,color = "black",label = "Predicted Salaries")
       plt.xlabel("Years of Expeirence")
       plt.ylabel("Salaries")
       plt.legend()
       plt.show()

       d = pd.DataFrame({"Experience":exp,"Salaries":salaries})
       print("\n",d)

    except FileNotFoundError:
        print("\nNo csv File Exist in The Directory")
        print("Press Enter Key to Exit")
        input()
        exit()



main()
#if __name__ == "__main()__":
   # main()
   # input()
