"""
Database Models for Heart Disease Detection System
This module defines the database structure and relationships for the application.

"""

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User

"""
Model: Consumer
Description: Represents a consumer/student in the system
Fields:
    - name: Student's full name
    - email: Student's email address
    - image: Optional profile image
    - content: Additional information about the consumer
"""
class Consumer(models.Model):
    """
    Consumer Model
    Represents a student or end-user of the system who will use the retinal analysis service.
    
    Fields:
    -------
    name : CharField
        Full name of the student/consumer
    email : EmailField
        Contact email address for communications
    image : ImageField
        Optional profile picture or identification image
    content : TextField
        Additional notes or information about the consumer
    """
    name = models.CharField(max_length=100, verbose_name="Student Name")
    email = models.EmailField(max_length=277, verbose_name="Student Email")
    image = models.ImageField(upload_to='consumer_images/', null=True, blank=True, verbose_name="Consumer Image")
    content = models.TextField(verbose_name="Consumer Content", blank=True, null=True)

    def __str__(self):
        return str(self.id)

from django.db import models
from django.contrib.auth.models import User

"""
Model: Profile
Description: Extended user profile model with additional fields
Fields:
    - user: One-to-one relationship with Django's User model
    - profile_picture: Optional profile image
"""
class Profile(models.Model):
    """
    Profile Model
    Extends the built-in Django User model with additional profile information.
    
    Fields:
    -------
    user : OneToOneField
        Link to Django's built-in User model (1:1 relationship)
    profile_picture : ImageField
        Optional profile picture for the user's account
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)

"""
Function: create_profile
Description: Signal handler that automatically creates a profile when a new user is created
Parameters:
    sender: The model class that sent the signal
    kwargs: Additional keyword arguments including the created instance
"""
def create_profile(sender, **kwargs):
    """
    Signal handler to automatically create a profile when a new user is registered.
    This ensures every user has an associated profile entry.
    
    Parameters:
    -----------
    sender : Model class
        The model class that sent the signal (User in this case)
    kwargs : dict
        Additional keyword arguments including the created instance
    """
    if kwargs['created']:
        Profile.objects.create(user=kwargs['instance'])

models.signals.post_save.connect(create_profile, sender=User)

"""
Model: Contact
Description: Stores contact form submissions
Fields:
    - name: Contact person's name
    - email: Contact email address
    - message: Contact message content
"""
class Contact(models.Model):
    """
    Contact Model
    Stores messages received through the contact form.
    
    Fields:
    -------
    name : CharField
        Name of the person making contact
    email : EmailField
        Email address for responding to the contact
    message : TextField
        Content of the contact message
    """
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()

    def __str__(self):
        return self.name

"""
Model: PredictionResult
Description: Stores results from retinal image analysis predictions
Fields:
    - age: Predicted age
    - sbp: Predicted systolic blood pressure
    - dbp: Predicted diastolic blood pressure
    - bmi: Predicted body mass index
    - hba1c: Predicted blood sugar level
    - true_*: Actual values for comparison (if available)
    - mae: Mean absolute error of predictions
    - risk_status: Overall health risk assessment
    - created_at: Timestamp of prediction
"""
class PredictionResult(models.Model):
    """
    PredictionResult Model
    Stores the results of retinal image analysis and predictions.
    
    Fields:
    -------
    Predicted Values:
        age : float
            Predicted age from retinal image
        sbp : float
            Predicted Systolic Blood Pressure
        dbp : float
            Predicted Diastolic Blood Pressure
        bmi : float
            Predicted Body Mass Index
        hba1c : float
            Predicted HbA1c (blood sugar) level
            
    True Values (for validation):
        true_age : float
            Actual age if known
        true_sbp : float
            Actual Systolic Blood Pressure if known
        true_dbp : float
            Actual Diastolic Blood Pressure if known
        true_bmi : float
            Actual BMI if known
        true_hba1c : float
            Actual HbA1c if known
            
    Analysis:
        mae : float
            Mean Absolute Error of predictions
        risk_status : str
            Overall health risk assessment (Low/Medium/High)
        created_at : DateTime
            Timestamp of when prediction was made
    """
    age = models.FloatField()
    sbp = models.FloatField()  # Systolic Blood Pressure
    dbp = models.FloatField()  # Diastolic Blood Pressure
    bmi = models.FloatField()  # Body Mass Index
    hba1c = models.FloatField()  # HbA1c (Blood Sugar)
    
    true_age = models.FloatField(null=True)
    true_sbp = models.FloatField(null=True)
    true_dbp = models.FloatField(null=True)
    true_bmi = models.FloatField(null=True)
    true_hba1c = models.FloatField(default=0.0)

    mae = models.FloatField(default=0.0)  # Mean Absolute Error
    risk_status = models.CharField(max_length=20, null=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True)

    def __str__(self):
        return f"Prediction at {self.created_at} - {self.risk_status}"

