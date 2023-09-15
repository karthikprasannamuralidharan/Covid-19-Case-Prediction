from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('main',views.main,name='main'),
    path('dashboard',views.dashboard,name='dashboard'),
    path('prediction',views.prediction,name='prediction'),
    path('analysis_ind',views.analysis_ind,name='analysis_ind'),
    path('analysis_st',views.analysis_st,name='analysis_st'),
    path('analysis_dt',views.analysis_dt,name='analysis_dt'),
    path('',views.home,name='home'),
]