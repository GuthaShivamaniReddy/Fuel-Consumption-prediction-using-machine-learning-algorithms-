from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class Driver_Profile_Classification(models.Model):

    sexofdriver= models.CharField(max_length=3000)
    agebandofdriver= models.CharField(max_length=3000)
    educationlevel= models.CharField(max_length=3000)
    vehicledriverrelation= models.CharField(max_length=3000)
    driverexperience= models.CharField(max_length=3000)
    typeofvehicle= models.CharField(max_length=3000)
    ownerofvehicle= models.CharField(max_length=3000)
    defectofvehicle= models.CharField(max_length=3000)
    roadsurfacecondition= models.CharField(max_length=3000)
    fuelconsumption= models.CharField(max_length=3000)   
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



