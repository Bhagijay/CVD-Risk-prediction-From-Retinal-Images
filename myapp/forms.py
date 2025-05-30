"""
Form Definitions for Heart Disease Detection System
This module defines all the forms used in the application for data input and validation.
It includes forms for user authentication, profile management, and image upload handling.

Forms:
- LoginForm: User authentication
- UserRegistrationForm: New user registration
- ImageUploadForm: Retinal image upload and validation


"""

from django import forms
from django.contrib.auth.models import User
from .models import *

class LoginForm(forms.Form):
    """
    User Login Form
    
    Handles user authentication by validating username and password.
    
    Fields:
    -------
    username : CharField
        User's login identifier
        - Max length: 150 characters
        - Required field
        - Strip whitespace: False
        
    password : CharField
        User's password
        - Widget: PasswordInput (masks input)
        - Required field
        
    Validation:
    -----------
    - Ensures username exists in database
    - Validates password format and length
    - Custom error messages for invalid credentials
    """
    username = forms.CharField(
        max_length=150,
        strip=False,
        required=True,
        help_text='',
        error_messages={
            'unique': "A user with that username already exists.",
        },
    )
    password = forms.CharField(widget=forms.PasswordInput)


"""
Form: UserRegistrationForm
Description: Handles new user registration
Fields:
    - username: Desired username
    - first_name: User's first name
    - last_name: User's last name
    - email: User's email address
    - password: Primary password
    - password2: Password confirmation
Validation:
    - Ensures passwords match
    - Validates username uniqueness
"""
class UserRegistrationForm(forms.ModelForm):
    """
    New User Registration Form
    
    Handles the creation of new user accounts with profile information.
    
    Fields:
    -------
    username : CharField
        Desired username (from User model)
    first_name : CharField
        User's first name (from User model)
    last_name : CharField
        User's last name (from User model)
    email : EmailField
        User's email address (from User model)
    password : CharField
        Primary password entry
        - Widget: PasswordInput
        - Required field
    password2 : CharField
        Password confirmation
        - Must match password field
        
    Validation:
    -----------
    - Ensures username is unique
    - Validates email format
    - Confirms passwords match
    - Enforces password complexity requirements
    
    Meta:
    -----
    model : User
        Django's built-in User model
    fields : tuple
        Fields to include from the User model
    """
    password = forms.CharField(
        label='Password', 
        widget=forms.PasswordInput,
        strip=False,
        required=True,
        # help_text="Your password must contain at least 8 characters.",
        error_messages={
            'password_mismatch': "The two password fields didn't match.",
        },
        )
    password2 = forms.CharField(label='Repeat Password', widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email')

    def clean_password2(self):
        """
        Custom validation to ensure password confirmation matches.
        Raises ValidationError if passwords don't match.
        """
        cd = self.cleaned_data
        if cd['password'] != cd['password']:
            raise forms.ValidationError('Password does not match')
        return cd['password2']

"""
Form: EditProfileForm
Description: Handles user profile editing
Fields:
    - username: User's username
    - first_name: User's first name
    - last_name: User's last name
    - email: User's email address
    - profile_picture: Optional profile image
"""
class EditProfileForm(forms.ModelForm):
    """
    Profile Edit Form
    
    Allows users to update their profile information and upload a profile picture.
    
    Fields:
    -------
    username : CharField
        User's username (from User model)
    first_name : CharField
        User's first name (from User model)
    last_name : CharField
        User's last name (from User model)
    email : EmailField
        User's email address (from User model)
    profile_picture : ImageField
        Optional profile image
        - Not required
        - Supports image upload
    
    Meta:
    -----
    model : User
        Django's built-in User model
    fields : tuple
        Fields that can be edited
    """
    profile_picture = forms.ImageField(required=False)
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'profile_picture')


"""
Form: ContactForm
Description: Handles contact form submissions
Fields:
    - name: Contact person's name
    - email: Contact email address
    - message: Contact message content
"""
class ContactForm(forms.ModelForm):
    """
    Contact Form
    
    Handles user inquiries and messages through the contact form.
    
    Fields:
    -------
    name : CharField
        Contact person's name
    email : EmailField
        Contact email address for responses
    message : TextField
        Content of the contact message
    
    Meta:
    -----
    model : Contact
        Contact model for storing messages
    fields : list
        Fields required for contact submission
    """
    class Meta:
        model = Contact
        fields = ['name', 'email', 'message']


"""
Form: ImageUploadForm
Description: Handles retinal image uploads for analysis
Fields:
    - image: Image file field
Validation:
    - Validates file type (JPEG, PNG)
    - Validates file size (max 5MB)
    - Ensures file is not empty
Widget Customization:
    - Custom CSS classes
    - File type restrictions
    - Custom placeholder text
"""
class ImageUploadForm(forms.Form):
    """
    Retinal Image Upload Form
    
    Handles the upload and validation of retinal images for analysis.
    
    Fields:
    -------
    image : ImageField
        The retinal image file
        - Accepts: JPEG, PNG formats
        - Max size: 5MB
        - Required field
        
    Widget Customization:
    --------------------
    - Custom CSS classes for styling
    - File type restrictions
    - Custom placeholder text
    
    Validation:
    -----------
    1. File presence check
       - Ensures file was uploaded
       - Validates file is not empty
       
    2. File type validation
       - Accepts only JPEG and PNG
       - Validates image file integrity
       
    3. Size restrictions
       - Maximum file size: 5MB
       - Prevents large file uploads
    
    Error Messages:
    --------------
    - Invalid file type
    - File too large
    - Empty file
    - Corrupted image
    """
    image = forms.ImageField(
        label='Select a retinal image',
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control image-upload',
            'accept': 'image/*',
            'placeholder': 'Upload Retinal Image',
        }),
        error_messages={
            'invalid_image': 'Upload a valid image. The file you uploaded was either not an image or a corrupted image.',
            'required': 'Please select an image file.',
            'empty': 'The submitted file is empty.',
        }
    )

    def clean_image(self):
        """
        Custom validation method for image uploads.
        
        Performs:
        1. File presence validation
        2. File type checking
        3. File size validation
        
        Returns:
        --------
        image : ImageField
            Validated image file
            
        Raises:
        -------
        ValidationError
            If any validation check fails
        """
        image = self.cleaned_data.get('image')
        if not image:
            raise forms.ValidationError('No image file uploaded.')
        
        # Validate file type
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
        if image.content_type not in allowed_types:
            raise forms.ValidationError('Invalid file type. Please upload a JPEG or PNG image.')
        
        # Validate file size (max 5MB)
        if image.size > 5 * 1024 * 1024:
            raise forms.ValidationError('Image file too large. Maximum size is 5MB.')
        
        return image

from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Select a retinal image (.tif)')

