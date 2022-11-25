import plotly.express as pe
import pandas as pd
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score 


data = pd.read_csv("project.csv")

toefl = data["TOEFL Score"].tolist()
gre = data["GRE Score"].tolist()
chanceadmit = data["Chance of admit"].tolist()


colors=[]


for i in chanceadmit:
    if i==1:
        colors.append("green")
    else:
        colors.append("red")

graph=go.Figure(data=go.Scatter(
    x=toefl,
    y=gre,
    mode='markers',
    marker=dict(color=colors)
))

graph.show()


scores = data[["GRE Score", "TOEFL Score"]]

results = data["Chance of admit"]

score_train, score_test, result_train, result_test = train_test_split(scores, results, test_size=0.25)


lr=LogisticRegression(random_state=0)
lr.fit(score_train, result_train)

results_pred = lr.predict(score_test)

print("The accuracy is : " , accuracy_score(result_test , results_pred))


from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 


user_gre_score = int(input("Enter the GRE scoret -> "))
user_toefl_score = int(input("Enter the TOEFL Score -> "))

user_test =  sc_x.transform([[user_gre_score, user_toefl_score]])

user_admission_pred = lr.predict(user_test)

if user_admission_pred[0] == 1:
  print("This student will recieve admission!")
else:
  print("This student will not recieve admission!")





