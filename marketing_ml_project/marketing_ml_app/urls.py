
from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
   path('about/',views.about),
   path('',views.about),
   path('dataframe/',views.combined_dataframe_view), 
   path('model_view/',views.model_view), 
   path('train_model/',views.train_model_view, name='train_model'),  
   path('model_analysis/', views.graphing, name='graphs'),
   path('predict_models/', views.predict_model_view),
   ]
