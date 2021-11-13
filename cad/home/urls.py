from django.urls import path
from . import views
from django.conf.urls import url, include

urlpatterns = [
        path('', views.index, name = 'index'), 
        path('form', views.form, name = 'form'), 
        url('^django_plotly_dash/', include('django_plotly_dash.urls')), 
]