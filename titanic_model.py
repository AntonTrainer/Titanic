import numpy as np 
import pandas as pd

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split #built in training data splitter
from sklearn.metrics import accuracy_score #built in sci kit tool to measure accuracy, try to build without this after the model is functioning 
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt 

test = pd.read_csv("test.csv")
train = pd.read_csv('train.csv')


#print(train["Pclass"].value_counts()) #check how many classes to avoid errors later
#print(train["Sex"].value_counts()) #check for male female, if no NaN vals, all good, if NaN values, delete or reassign 

#-------------------------------------- data processing --------------------------------------------------------------------------------
def age_cats(df, cutoff, assignment):
    df['Age'] = df['Age'].fillna(-.5) #set all the missing age values to -.5 
    df['Age_group'] = pd.cut(df['Age'], cutoff, labels = assignment) #Cut the age column into 6 categories, tried more categories but it lowered accuracy 
    return df #build out dataframe with the new assignments, making a new header for each 
cutoff = [-1,0,5,12,18,35,60,100] #define the cuts 
assignment = [0,1,2,3,4,5,6] #assign the cuts 

train = age_cats(train, cutoff, assignment) 
test = age_cats(test, cutoff, assignment) 

test.replace(["Q", "C", "S"], [0, 1, 2], inplace=True) #re-assign embark str values to int values  
train.replace(["Q", "C", "S"], [0, 1, 2], inplace=True) #re-assign embark str values to int values 

''' #tried to use SibSp to get better accuracy but it lowered acc 
def sibpar(df, cutoff, assignment): #sib/sp cut offs
    df['SibSp'] = df['SibSp'].fillna(0)
    df['sib'] = pd.cut(df['SibSp'], cutoff, labels = assignment)
    return df
cutoff = [-1,0,1,2,3,4,5,6,10]
assignment = [0,1,2,3,4,5,6,7]
'''

def create_holders(df, column_name): #need to make each persons Pclass a list with 3 indexes and one will be a 1 to represent which class they were and the rest 0's
    place_holder = pd.get_dummies(df[column_name],prefix=column_name) #builds out the place holder lists when given a header
    df = pd.concat([df,place_holder], axis=1) #concenate the original df and the new place_holder
    return df 

for col in ["Pclass", 'Sex', 'Age_group', 'Embarked', 'SibSp', 'Parch']:
    train =create_holders(train,col) #runs the columns through create_holders
    test = create_holders(test,col) #runs through the test bs 
    
train = train.drop(['Name', 'Ticket', 'Cabin', 'Age', 'Sex'], axis =1) #drop the meaningless data and onehot columns 
test = test.drop(['Name', 'Ticket', 'Cabin', 'Age','Sex'], axis =1) #drop the meaningless data and onehot columns 
#print(test) #check the data to endure evverything looks OK, not necessary to keep
'''
df = test 
fig, ax1 = plt.subplots(1,1)
sb.heatmap(df, ax=ax1, alpha=0.1)'''

#------------------------------------------------Splitting the Data -------------------------------------------------------------------------
#Splitting Training data to train and test then check accuracy  

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male','Age_group_0','Age_group_1','Age_group_2','Age_group_3','Age_group_4',
'Age_group_5','Age_group_6', 'Embarked_0.0', 'Embarked_1.0', 'Embarked_2.0']#need to know the columns for the x variables of the predict
#extra values that didn't help acc 'SibSp_0' , 'SibSp_1' , 'SibSp_2' , 'SibSp_3' , 'SibSp_4' , 'SibSp_5', 'SibSp_8','Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Parch_9']'''

lr = LogisticRegression()
lr.fit(train[columns], train['Survived'])
#print(lr)

holdout_data = test #change the test data from kaggle to holdout to use seperately from the new train data which will be split and randomized from the original train data 

all_x = train[columns]
all_y = train['Survived']

train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=.18, random_state=20) #test size is the proportion of the train data you want to test on, !!set random state to an interger after testing

#------------------------------------- Running the model and producing accuracy measures on train data -------------------------------------------------

lr.fit(train_x,train_y) #trains on the predefined proportion 
predictions = lr.predict(test_x) #only needs the x variable when predicting, should return a single dimensional array of predictions for the test portion of the train data 
#print(predictions) #printed just to check if the random_state changed the output
scores = cross_val_score(lr, all_x, all_y, cv=12) #see notes below
scores.sort()
accuracy = accuracy_score(test_y,predictions) #sci kit built in to test the predictions with the y_test proportion of the training data see notes below from scikit documentation
print("Accuracy of the model: " + str(accuracy*100) +'%')
print("Validation Scores: "+ str(scores))
print(np.mean(scores))

#---------------------------------- Running on unseen 'holdout data'-----------------------------------------------------------
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male','Age_group_0','Age_group_1','Age_group_2','Age_group_3','Age_group_4','Age_group_5','Age_group_6',
 'Embarked_0', 'Embarked_1', 'Embarked_2'] #Again, can add these but hurts acc 'SibSp_0' , 'SibSp_1' , 'SibSp_2' , 'SibSp_3' , 'SibSp_4' , 'SibSp_5', 'SibSp_8''Parch_0'
lr.fit(all_x,all_y)
holdout_predictions = lr.predict(holdout_data[columns])

print(holdout_predictions)

#----------------------------------Creating a submission file--------------------------------------------------------------------

''' "The file should have 2 columns, a Passenger ID and a surived, this should have 418 rows plus the header to fit the Kaggle guielines for a submission file."''' #from kaggle guidelines

holdout_ids = holdout_data['PassengerId']
submission_df = {"PassengerId" : holdout_ids, "Survived":holdout_predictions} # build a dict to hold the pass Id with thier suvival verdict, the order wont change for either  
submission = pd.DataFrame(submission_df) #build out a pandas dataframe 

submission.to_csv("submission.csv", index=False) #need to have the index set to false or it will add a column to the csv 

#-------------------------------------------------Notes-------------------------------------------------------------

''' using K-Fold cross validation method to train on several splits of data and then average the accuracy scores so it isn't over fitting
a fold is a single itteration the it trains on, k is the number of folds. '''



'''estimator is a scikit-learn estimator object, like the LogisticRegression() objects we have been creating.
X is all features from data set.
y is the target variables.
cv specifies the number of folds.'''

#tried to use train test split to build out the portion to train/test on but scikit train_test_split is neater 
'''
def ttSplit(x, y): #tt split because splitting testing and training data, all of the x's and all of the y's, the pct is the ratio of the training data to testing data that we watn
    pct = .85 # tweeakable
    numtest = int(len(y)*pct)   #the index that sets the pivot between training and testing 

    # x_train, y_train, x_test, y_test
    return x[:numtest], y[:numtest], x[numtest:], y[numtest:] '''
