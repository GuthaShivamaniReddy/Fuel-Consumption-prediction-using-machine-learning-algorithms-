
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
 
# Create your views here.
from Remote_User.models import ClientRegister_Model,Driver_Profile_Classification,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def View_Predicted_Driver_Profile_Classification_Type_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Less and Good Behaviour'
    print(kword)
    obj = Driver_Profile_Classification.objects.all().filter(Q(Prediction=kword))
    obj1 = Driver_Profile_Classification.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio12 = ""
    kword12 = 'More and Bad Behaviour'
    print(kword12)
    obj12 = Driver_Profile_Classification.objects.all().filter(Q(Prediction=kword12))
    obj112 = Driver_Profile_Classification.objects.all()
    count12 = obj12.count();
    count112 = obj112.count();
    ratio12 = (count12 / count112) * 100
    if ratio12 != 0:
        detection_ratio.objects.create(names=kword12, ratio=ratio12)

    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Predicted_Driver_Profile_Classification_Type_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Predicted_Driver_Profile_Classification_Type(request):
    obj =Driver_Profile_Classification.objects.all()
    return render(request, 'SProvider/View_Predicted_Driver_Profile_Classification_Type.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Predicted_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Datasets.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Driver_Profile_Classification.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.sexofdriver, font_style)
        ws.write(row_num, 1, my_row.agebandofdriver, font_style)
        ws.write(row_num, 2, my_row.educationlevel, font_style)
        ws.write(row_num, 3, my_row.vehicledriverrelation, font_style)
        ws.write(row_num, 4, my_row.driverexperience, font_style)
        ws.write(row_num, 5, my_row.typeofvehicle, font_style)
        ws.write(row_num, 6, my_row.ownerofvehicle, font_style)
        ws.write(row_num, 7, my_row.defectofvehicle, font_style)
        ws.write(row_num, 8, my_row.roadsurfacecondition, font_style)
        ws.write(row_num, 9, my_row.fuelconsumption, font_style)        
        ws.write(row_num, 10, my_row.Prediction, font_style)


    wb.save(response)
    return response

def train_model(request):
    detection_accuracy.objects.all().delete()

    models = []
    file_path = 'fuel_consumption_classification.csv'
    df = pd.read_csv(file_path)

        # Encode categorical features
    label_encoders = {}
    for column in df.columns[:-2]:  # All columns except the last (label) and fuelconsumption
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

        # Split dataset into features and labels
    X = df.drop("label", axis=1)
    y = df["label"]

        # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    # SVM Model
    print("SVM")
    from sklearn import svm
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    models.append(('svm', lin_clf))
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

    print("Logistic Regression")

    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, y_pred) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))
    models.append(('logistic', reg))
    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

    print("GaussianNB")
    gb_model = GaussianNB()
    gb_model.fit(X_train, y_train)
    gppredict = gb_model.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, gppredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, gppredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, gppredict))
    models.append(('GaussianNB', gb_model))
    detection_accuracy.objects.create(names="GaussianNB", ratio=accuracy_score(y_test, gppredict) * 100)

    print("KNeighborsClassifier")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knnpredict = knn_model.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, knnpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, knnpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, knnpredict))
    models.append(('KNeighborsClassifier', knn_model))
    detection_accuracy.objects.create(names="KNeighborsClassifier", ratio=accuracy_score(y_test, knnpredict) * 100)



    csv_format = 'Results.csv'
    df.to_csv(csv_format, index=False)
    df.to_markdown

    obj = detection_accuracy.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})