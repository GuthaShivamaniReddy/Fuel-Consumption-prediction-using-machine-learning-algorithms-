from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

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
from django.contrib import messages
from django.shortcuts import render, redirect

def register(request):
    if request.method == "POST":
        # Assuming you process the form here
        username = request.POST.get("username")
        email = request.POST.get("email")
        
        # Add your form processing logic
        if username and email:  # Replace this with your actual logic
            # Add a success message
            messages.success(request, "Registration successful! Welcome, {}.".format(username))
            return redirect("register")  # Redirect to the same page to show the message
        else:
            # Add an error message
            messages.error(request, "Error during registration. Please try again.")
    
    return render(request, "register.html")


def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Driver_Profile_Classification(request):
    if request.method == "POST":

        if request.method == "POST":

            sexofdriver= request.POST.get('sexofdriver')
            agebandofdriver= request.POST.get('agebandofdriver')
            educationlevel= request.POST.get('educationlevel')
            vehicledriverrelation= request.POST.get('vehicledriverrelation')
            driverexperience= request.POST.get('driverexperience')
            typeofvehicle= request.POST.get('typeofvehicle')
            ownerofvehicle= request.POST.get('ownerofvehicle')
            defectofvehicle= request.POST.get('defectofvehicle')
            roadsurfacecondition= request.POST.get('roadsurfacecondition')
            fuelconsumption= request.POST.get('fuelconsumption')
            

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
        print("ACCURACY")
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

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

        # Input data to predict
        input_data = [sexofdriver,agebandofdriver, educationlevel, vehicledriverrelation,driverexperience, typeofvehicle, ownerofvehicle, defectofvehicle, roadsurfacecondition, fuelconsumption]

        # Encode the input data using the label encoders
        input_encoded = []
        for i, column in enumerate(df.columns[:-2]):  # Encode all categorical columns
            input_encoded.append(label_encoders[column].transform([input_data[i]])[0])
        input_encoded.append(input_data[-1])  # Append the numeric fuelconsumption value
        input_encoded = np.array(input_encoded).reshape(1, -1)

         

        prediction = knn_model.predict(input_encoded)
        print("ddddddddddddddddddddddddddddddd")
        print(prediction[0])
        predicted_class=''
        if prediction[0] ==0:
            predicted_class="Less and Good Behaviour"
        if prediction[0] ==1:
            print("tttttttttttttttttttttttttttttt")
            predicted_class="More and Bad Behaviour"

        print(predicted_class)
        Driver_Profile_Classification.objects.create(
        sexofdriver=sexofdriver,
        agebandofdriver=agebandofdriver,
        educationlevel=educationlevel,
        vehicledriverrelation=vehicledriverrelation,
        driverexperience=driverexperience,
        typeofvehicle=typeofvehicle,
        ownerofvehicle=ownerofvehicle,
        defectofvehicle=defectofvehicle,
        roadsurfacecondition=roadsurfacecondition,
        fuelconsumption=fuelconsumption,        
        Prediction=predicted_class)

        return render(request, 'RUser/Predict_Driver_Profile_Classification.html',{'objs': predicted_class})
    return render(request, 'RUser/Predict_Driver_Profile_Classification.html')



