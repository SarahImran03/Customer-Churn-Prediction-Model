import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

data_set = pd.read_csv("churn-bigml-80.csv")

le = LabelEncoder()
data_set['International plan'] = le.fit_transform(data_set['International plan'])
data_set['Voice mail plan'] = le.fit_transform(data_set['Voice mail plan'])
data_set['Churn'] = le.fit_transform(data_set['Churn'])
data_set['State'] = le.fit_transform(data_set['State'])

x = data_set.drop(['Churn'], axis=1)
y = data_set['Churn']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
train_data = train_x.join(train_y)

# Histograms of all the features
feature_names = x.columns.tolist()
hgram = pd.DataFrame(x)
hgram.columns = feature_names
fig, ax = mlib.subplots(figsize=(10,8))
hgram.hist(ax=ax)
mlib.subplots_adjust(hspace=0.571)
mlib.tight_layout()

# Heatmap of train_data
mlib.figure(figsize=(12,10))
sb.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')

# Pie chart of Churn in train_data
mlib.figure(figsize=(8,6))
train_data['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%')
mlib.title("Distribution of Churn")

mlib.show()

# Logistic Regression
log_regressor = LogisticRegression()
log_regressor.fit(train_x, train_y)

predicted_churn = log_regressor.predict(test_x)
accuracy = accuracy_score(test_y, predicted_churn)
precision = precision_score(test_y, predicted_churn)
recall = recall_score(test_y, predicted_churn)
f1 = f1_score(test_y, predicted_churn)

# Printing the scores as a Data Frame
result_data_frame = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                  'Value': [accuracy, precision, recall, f1]})
print(result_data_frame)

conf_matrix = confusion_matrix(test_y, predicted_churn)
print("\nThe confusion matrix:\n", conf_matrix)

cl_report = classification_report(test_y, predicted_churn)
print("\nThe classification report:\n", cl_report)