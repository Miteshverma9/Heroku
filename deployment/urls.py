from django.contrib import admin
from django.urls import path,include

urlpatterns = [
    path('', include('ML_Model.urls')),
    path("admin/", admin.site.urls),
]
