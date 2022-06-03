import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import seaborn as sns
import warnings
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm

keras = tf.keras
warnings.filterwarnings('ignore')

# READ ME
'''''''''
age (in years)
sex (1 = MALE, 0 = FEMALE)
chest pain type (4 values)
resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
serum cholesterol in mg/dl
fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
resting electrocardiographic results (values 0,1,2) 
maximum heart rate achieved
exercise induced angina (1 = yes; 0 = no)
oldpeak = ST depression induced by exercise relative to rest
the slope of the peak exercise ST segment
number of major vessels (0-3) colored by flourosopy
thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
'''''''''

# Reading the data
print(os.listdir(r'C:/Users/ata-d/'))
data = pd.read_csv(r'C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/heart.csv')

data.info()

# ----------------------------------------------------------------------------------------------------------------------

# DISTRIBUTIONS

# Histogram of the Ages by the Heart Disease (1 = Disease, 0 = Healthy)
'''''''''
plt.figure(figsize=(15, 8))

plt.hist([data[data['target'] == 0]['age'], data[data['target'] == 1]['age']], 48,
         stacked=True, density=True, alpha=0.75, color=['g', 'r'])
plt.xlabel("Ages", fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title("Histogram of the Ages by the Heart Disease", fontsize=16)
plt.legend(['Healthy', 'Disease'], loc='upper right')
plt.show()
'''''''''

# Distribution of the Disease vs Sex
'''''''''
plt.figure(figsize=(15, 8))
splot = sns.countplot(data=data, x='sex',
                      hue='target',
                      order=data['sex'].value_counts().index,
                      palette=['forestgreen', 'red'],
                      edgecolor=(0, 0, 0),
                      linewidth=2)

splot.set_xticklabels(['Male', 'Female'])

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 9),
                   textcoords='offset points')
plt.legend(['Healthy', 'Disease'], loc='upper right')
plt.ylabel('Frequency of the Disease', fontsize=14)
plt.xlabel('Sex', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Distribution of the Disease vs Sex', fontsize=20)
'''''''''

# Distribution of the Disease vs Chest Pain Type
'''''''''
plt.figure(figsize=(15, 8))
splot = sns.countplot(data=data, x='cp',
                      hue='target',
                      order=data['cp'].value_counts().index,
                      palette=['forestgreen', 'red'],
                      edgecolor=(0, 0, 0),
                      linewidth=2)

splot.set_xticklabels(['Asymptomatic', 'Without Relation to Angina', 'Atypical Angina',
                       'Typical Angina'])

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 9),
                   textcoords='offset points')

plt.legend(['Healthy', 'Disease'], loc='upper right')
plt.ylabel('Frequency of the Disease', fontsize=14)
plt.xlabel('Chest Pain Types', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Distribution of the Disease vs Chest Pain Type', fontsize=20)
'''''''''

# Histogram of the Resting Blood Pressure by the Heart Disease
'''''''''
plt.figure(figsize=(15, 8))

plt.hist([data[data['target'] == 0]['trestbps'], data[data['target'] == 1]['trestbps']], 20,
         stacked=True, density=True, alpha=0.75, color=['g', 'r'])
plt.xlabel("Resting Blood Pressure (mm Hg)", fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title("Histogram of the Resting Blood Pressure by the Heart Disease", fontsize=16)
plt.legend(['Healthy', 'Disease'], loc='upper right')
plt.show()
'''''''''

# Histogram of the Cholesterol by the Heart Disease
'''''''''
plt.figure(figsize=(15, 8))

plt.hist([data[data['target'] == 0]['chol'], data[data['target'] == 1]['chol']], 55,
         stacked=True, density=True, alpha=0.75, color=['g', 'r'])
plt.xlabel("Cholesterol (mg/dl)", fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title("Histogram of the Cholesterol by the Heart Disease", fontsize=16)
plt.legend(['Healthy', 'Disease'], loc='upper right')
plt.show()
'''''''''

# Heatmap of the FBS by the Heart Disease (0 = FBS is less than 120, 1 = FBS is greater than 120)
'''''''''
x = ['FBS is less than 120', 'FBS is greater than 120']
y = ['Disease', 'Healthy']
z = [[
      data[data['target'] == 1]['fbs'].value_counts()[0],
      data[data['target'] == 1]['fbs'].value_counts()[1]],
     [data[data['target'] == 0]['fbs'].value_counts()[0],
      data[data['target'] == 0]['fbs'].value_counts()[1]]
     ]

fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='magma')
fig.update_layout(title_text='Heatmap of the FBS by the Heart Disease',
                  title_x=0.5, title_font=dict(size=22))
fig.update_layout(xaxis=dict(
    tickfont=dict(size=15),
),
    yaxis=dict(tickfont=dict(size=15)))
fig.show()
'''''''''

# Distribution of the Resting Electrocardiography Results vs Heart Disease
'''''''''
plt.figure(figsize=(15, 8))
splot = sns.countplot(data=data, x='restecg',
                      hue='target',
                      order=data['restecg'].value_counts().index,
                      palette=['forestgreen', 'red'],
                      edgecolor=(0, 0, 0),
                      linewidth=2)

splot.set_xticklabels(['Normal', 'Hypertrophy', 'Abnormalities'])

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 9),
                   textcoords='offset points')
plt.legend(['Healthy', 'Disease'], loc='upper right')
plt.ylabel('Frequency of the Disease', fontsize=14)
plt.xlabel('Results of the Electrocardiogram', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Distribution of the Results of the Electrocardiogram on Rest vs Heart Disease', fontsize=20)
'''''''''

# Histogram of the Maximum Heart Rate by the Heart Disease
'''''''''
plt.figure(figsize=(15, 8))
plt.hist([data[data['target'] == 0]['thalach'], data[data['target'] == 1]['thalach']], 30,
         stacked=True, density=True, alpha=0.75, color=['g', 'r'])
plt.xlabel("Maximum Heart Rates", fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title("Histogram of the Maximum Heart Rate by the Heart Disease", fontsize=16)
plt.legend(['Healthy', 'Disease'], loc='upper right')
plt.show()
'''''''''

# Heatmap of the Exercise Induced Angina (1 = yes; 0 = no)
'''''''''
x = ['No', 'Yes']
y = ['Disease', 'Healthy']
z = [[
      data[data['target'] == 1]['exang'].value_counts()[0],
      data[data['target'] == 1]['exang'].value_counts()[1]],
     [data[data['target'] == 0]['exang'].value_counts()[0],
      data[data['target'] == 0]['exang'].value_counts()[1]]
     ]

fig = ff.create_annotated_heatmap(z, x=x, y=y)
fig.update_layout(title_text='Heatmap of the Exercise Induced Angina by the Heart Disease',
                  title_x=0.5, title_font=dict(size=22))
fig.update_layout(xaxis=dict(
    tickfont=dict(size=15),
),
    yaxis=dict(tickfont=dict(size=15)))
fig.show()
'''''''''

# Distribution of the ST Depression Induced by Exercise Relative to Rest
'''''''''
plt.figure(figsize=(15, 8))
sns.distplot(data.loc[data['target'] == 0][['oldpeak']], hist=True)
sns.distplot(data.loc[data['target'] == 1][['oldpeak']], hist=True)
plt.xlabel("ST Depression Induced by Exercise Relative to Rest", fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(['Healthy', 'Disease'], loc='upper right')

plt.title("Density Plot of the ST Depression Induced by Exercise Relative to Rest", fontsize=16)
plt.show()
'''''''''

# Distribution of the Slope of the ST Segment
'''''''''
plt.figure(figsize=(15, 8))
splot = sns.countplot(data=data, x='slope',
                      hue='target',
                      order=data['slope'].value_counts().index,
                      palette=['forestgreen', 'red'],
                      edgecolor=(0, 0, 0),
                      linewidth=2)

splot.set_xticklabels(['Ascending', 'Flat', 'Descending'])

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 9),
                   textcoords='offset points')
plt.legend(['Healthy', 'Disease'], loc='upper right')
plt.ylabel('Frequency of the Disease', fontsize=14)
plt.xlabel('Slopes', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Distribution of the Slope of the ST Segment by Heart Disease', fontsize=20)
'''''''''

# Distribution of the Number of Major Vessels Colored by Fluoroscopy
'''''''''
fig = go.Figure(data=[
    go.Bar(name='Healthy', x=data[data['target'] == 0]['ca'].value_counts().index,
           y=data[data['target'] == 0]['ca'].value_counts(),
           marker=dict(color="green", line=dict(width=5), opacity=0.78)),

    go.Bar(name='Disease', x=data[data['target'] == 1]['ca'].value_counts().index,
           y=data[data['target'] == 1]['ca'].value_counts(),
           marker=dict(color="red", line=dict(width=5), opacity=0.78)
           )
])

fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})

fig.update_xaxes(title_text="Major Vessels Colored by Fluoroscopy", title_font={"size": 16})
fig.update_yaxes(title_text="Frequency", title_font={"size": 16})

fig.update_layout(title_text='Distribution of the Number of Major Vessels Colored by Flourosopy',
                  title_x=0.5, title_font=dict(size=20))
fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))

fig.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# Data Pre-Process

X = data.drop('target', axis=1)
y = data['target']

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=13)

rf = RandomForestClassifier(max_depth=50, max_features=6, max_leaf_nodes=20,
                            min_samples_leaf=4, min_samples_split=8,
                            n_estimators=20,
                            bootstrap=True)

rf.fit(trainX, trainY)

train_scoreRF = rf.score(trainX, trainY)
test_scoreRF = rf.score(testX, testY)

print('RF Train Score: %', train_scoreRF * 100)
print('RF Test Score: %', test_scoreRF * 100)

# ----------------------------------------------------------------------------------------------------------------------

# SVM Classification

clf = svm.SVC(kernel='linear')
clf.fit(trainX, trainY)

predictions_svm = clf.predict(testX)

train_scoreSVM = clf.score(trainX, trainY)
test_scoreSVM = clf.score(testX, testY)

print('SVM Train Score: %', train_scoreSVM * 100)
print('SVM Test Score: %', test_scoreSVM * 100)

# ----------------------------------------------------------------------------------------------------------------------

# XGBClassifier

model = XGBClassifier(learning_rate=0.001, max_depth=10, n_estimators=30,
                      colsample_bytree=0.3, min_child_weight=0.4, reg_alpha=0.1,
                      )
model.fit(trainX, trainY)

train_scoreXGB = model.score(trainX, trainY)
test_scoreXGB = model.score(testX, testY)

print('XGBClassifier Train Score: %', train_scoreXGB * 100)
print('XGBClassifier Test Score: %', test_scoreXGB * 100)

# ----------------------------------------------------------------------------------------------------------------------

# Logistic Regression

reggressor = LogisticRegression(random_state=13, max_iter=2000).fit(trainX, trainY)

train_scoreLR = reggressor.score(trainX, trainY)
test_scoreLR = reggressor.score(testX, testY)

print('LR Train Score: %', train_scoreLR * 100)
print('LR Test Score: %', test_scoreLR * 100)

# ----------------------------------------------------------------------------------------------------------------------

train_list = [train_scoreRF, train_scoreSVM, train_scoreXGB, train_scoreLR]
test_list = [test_scoreRF, test_scoreSVM, test_scoreXGB, test_scoreLR]
names = ['RandomForest', 'SVM', 'XGBClassifier', 'Logistic Regression']

train_list = pd.DataFrame(train_list)
test_list = pd.DataFrame(test_list)
names = pd.DataFrame(names)

train_list.columns = ['train']
test_list.columns = ['test']
names.columns = ['names']


over_all_score = pd.concat([names, train_list, test_list], axis=1)

# ----------------------------------------------------------------------------------------------------------------------


fig = go.Figure(data=[
    go.Bar(name='Train Results', x=over_all_score['names'], y=over_all_score['train'],
           marker=dict(line=dict(width=5)),
           texttemplate='%{y:20,.2f}', textposition='outside',),

    go.Bar(name='Test Results', x=over_all_score['names'], y=over_all_score['test'],
           marker=dict(line=dict(width=5)),
           texttemplate='%{y:20,.2f}', textposition='outside',)
])

fig.update_layout(barmode='group', xaxis={'categoryorder': 'total descending'})

fig.update_xaxes(title_text="Models", title_font={"size": 16})
fig.update_yaxes(title_text="Error Rate", title_font={"size": 16})

fig.update_layout(title_text='Train and Test Results of the each Model',
                  title_x=0.5, title_font=dict(size=20))
fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))

fig.show()


# ----------------------------------------------------------------------------------------------------------------------
