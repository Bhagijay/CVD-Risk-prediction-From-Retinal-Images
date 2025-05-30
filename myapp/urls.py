"""
URL Configuration for Heart Disease Detection System
This module defines all URL patterns and their corresponding view functions.
It maps URLs to views and provides named URL patterns for reverse lookup.

URL Categories:
- Authentication URLs (login, logout, password management)
- User Management URLs (registration, profile)
- Application Features (prediction, dataset, history)
- Static/Media File Serving
"""

from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

from django.conf import settings
from django.conf.urls.static import static

# Main URL patterns for the application
urlpatterns = [
    # Basic Pages
    path('', views.base, name='base'),  # Homepage
    path('about/', views.about, name='about'),  # About page
    path('dashboard/', views.dashboard, name='dashboard'),  # User dashboard
    
    # Authentication URLs
    path('login/', auth_views.LoginView.as_view(), name='login'),  # User login
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),  # User logout

    # Password Management URLs
    path('password_change/', 
         auth_views.PasswordChangeView.as_view(), 
         name='password_change'),  # Change password form
         
    path('password_change/done/', 
         auth_views.PasswordChangeDoneView.as_view(), 
         name='password_change_done'),  # Password change success

    # Password Reset Flow
    path('password_reset/', 
         auth_views.PasswordResetView.as_view(
            template_name='registration/password_reset_form.html',
            email_template_name='registration/password_reset_email.html',
            subject_template_name='registration/password_reset_subject.txt'
         ), 
         name='password_reset'),  # Initialize password reset

    path('password_reset/done/', 
         auth_views.PasswordResetDoneView.as_view(
            template_name='registration/password_reset_done.html'
         ), 
         name='password_reset_done'),  # Password reset email sent

    path('reset/<uidb64>/<token>/', 
         auth_views.PasswordResetConfirmView.as_view(
            template_name='registration/password_reset_confirm.html'
         ), 
         name='password_reset_confirm'),  # Reset password form

    path('reset/done/', 
         auth_views.PasswordResetCompleteView.as_view(
            template_name='registration/password_reset_complete.html'
         ), 
         name='password_reset_complete'),  # Password reset success

    # User Registration and Profile Management
    path('register/', views.register, name='register'),  # New user registration
    path('profile/', views.profile, name='profile'),  # User profile view
    path('edit-profile/', views.edit_profile, name='edit_profile'),  # Edit profile
    path('delete-account/', views.delete_account, name='delete_account'),  # Delete account

    # Contact Form
    path('contact/', views.contact, name='contact'),  # Contact form

    # Application Features
    path('predict/', views.predict, name='predict'),  # Retinal image prediction
    path('dataset/', views.dataset, name='dataset'),  # Dataset management
    path('history/', views.prediction_history, name='prediction_history'),  # View prediction history
    path('detectHeartDisease/', 
         views.heart_disease_detection_view, 
         name='detect_heart_disease'),  # Heart disease detection
]

# Serve media files in development
# In production, these should be served by the web server
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

