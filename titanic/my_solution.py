import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. read data
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
combine = [data_train,data_test]

# 2. deal with data
# 2.1 sex
for dataset in combine:
	dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)

# 2.2 age 
guess_ages = np.zeros((2,3))
for dataset in combine:
	for i in range(0,2):
		for j in range(0,3):
			guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
			age_guess = guess_df.median()
			guess_ages[i,j] = int(age_guess / 0.5 + 0.5) * 0.5

	for i in range(0,2):
		for j in range(0,3):
			dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[i,j]
	dataset['Age'] = dataset['Age'].astype(int)

# 2.3 embarked
freq_port = 'S'
for dataset in combine:
	freq_port = dataset['Embarked'].dropna().mode()[0]
for dataset in combine:
	dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
	dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

# 2.4 fare
data_test['Fare'].fillna(data_test['Fare'].dropna().median(), inplace=True)
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

# 2.5 cabin
for dataset in combine:
	dataset.loc[(dataset.Cabin.isnull()), 'Cabin'] = 0
	dataset.loc[(dataset.Cabin.notnull()),'Cabin'] = 1

# 3. prepare for the data
X_train = data_train[['Pclass','Sex','Age','Fare','Embarked','Cabin']]
Y_train = data_train['Survived']
PassengerId = data_test['PassengerId']
X_test  = data_test[['Pclass','Sex','Age','Fare','Embarked','Cabin']]

# 3.1 heatmap
# colormap = plt.cm.RdBu
# plt.figure(figsize=(8,6))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(X_train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
#             square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()

# 4. compare with different models
# 4.1 logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
prediction_logreg = logreg.predict(X_test)
accuracy_logreg = round(logreg.score(X_train,Y_train) * 100 , 2)

# 4.2 support vector machines
svc = SVC()
svc.fit(X_train, Y_train)
prediction_svc = svc.predict(X_test)
accuracy_svc = round(svc.score(X_train,Y_train) * 100 , 2)

# 4.3 k-nearest neighbors
knn =  KNeighborsClassifier()
knn.fit(X_train, Y_train)
prediction_knn = knn.predict(X_test)
accuracy_knn = round(knn.score(X_train,Y_train) * 100 , 2)

# 4.4 gaussian naive bayes
gaussian =  GaussianNB()
gaussian.fit(X_train, Y_train)
prediction_gaussian = gaussian.predict(X_test)
accuracy_gaussian = round(gaussian.score(X_train,Y_train) * 100 , 2)

# 4.5 linear svc
linear_svc =  LinearSVC()
linear_svc.fit(X_train, Y_train)
prediction_linear_svc = linear_svc.predict(X_test)
accuracy_linear_svc = round(linear_svc.score(X_train,Y_train) * 100 , 2)

# 4.6 stochastic gradient descent
sgd =  SGDClassifier()
sgd.fit(X_train, Y_train)
prediction_sgd = sgd.predict(X_test)
accuracy_sgd = round(sgd.score(X_train,Y_train) * 100 , 2)

# 4.7 decision tree
dtc =  DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
prediction_dtc = dtc.predict(X_test)
accuracy_dtc = round(dtc.score(X_train,Y_train) * 100 , 2)

# 4.8 random forest
random_forest =  RandomForestClassifier()
random_forest.fit(X_train, Y_train)
prediction_random_forest = random_forest.predict(X_test)
accuracy_random_forest = round(random_forest.score(X_train,Y_train) * 100 , 2)

models = pd.DataFrame({
	'Model':['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score':[accuracy_svc, accuracy_knn,accuracy_logreg,
    		 accuracy_random_forest,accuracy_gaussian,
    		 accuracy_sgd,accuracy_linear_svc,
    		 accuracy_dtc]
	})

print(models.sort_values(by = 'Score', ascending = False) )
#                         Model  Score
# 7               Decision Tree  92.82
# 3               Random Forest  91.92
# 0     Support Vector Machines  85.97
# 1                         KNN  84.62
# 2         Logistic Regression  79.69
# 6                  Linear SVC  79.46
# 4                 Naive Bayes  75.98
# 5  Stochastic Gradient Decent  63.30

# 5. model selection
# as the result above, the decison tree model is the best
# result = prediction_dtc

# # 6.write to the file
# submission = pd.DataFrame({
# 	"PassengerId":PassengerId,
# 	"Survived":result
# 	})
# submission.to_csv('submission_dtc0.csv', index = False)

# submission2 = pd.DataFrame({
# 	"PassengerId":PassengerId,
# 	"Survived":prediction_random_forest
# 	})
# submission.to_csv('submission_random_forest0.csv', index = False)

# question 1. in fact, accuracy is 73%, which should be 92%
# maybe overfitting

# from sklearn.model_selection import learning_curve
# train_sizes,train_loss,test_loss = learning_curve(
# 	DecisionTreeClassifier(), X_train,Y_train,cv = 5, scoring = 'neg_mean_squared_error',
# 	train_sizes = [0.1,0.25,0.5,0.75,1])
# train_loss_mean = -np.mean(train_loss,axis = 1)
# test_loss_mean  = -np.mean(test_loss, axis = 1)
# plt.plot(train_sizes, train_loss_mean,'o-',color = 'r',label = 'Training')
# plt.plot(train_sizes, test_loss_mean,'o-',color = 'g',label = 'Cross_validation')

# plt.xlabel('training examples')
# plt.ylabel('loss')
# plt.legend(loc = 'best')
# plt.show()