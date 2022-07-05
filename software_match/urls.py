"""software_match URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from first_web.views import *
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    # url(r'^index', index)
    path('', index),
    path('index/', index),
    path('single_prediction/', singlePrediction),
    path('single_prediction_result/', singlePredictionResult),
    path('single_prediction_save/', singlePredictionSave),
    path('show_list/', showList),
    path('batch_prediction/', batchPrediction),
    path('batch_prediction/batch_prediction_result/', batchPredictionResult)
]
urlpatterns += staticfiles_urlpatterns()
