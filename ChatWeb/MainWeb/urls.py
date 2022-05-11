from django.contrib import admin
from django.urls    import path, include
from MainWeb        import views


urlpatterns = [
    path('index/', views.index, name = 'index'),
    path('test/', views.test, name='test'),
    path('chatanswer', views.chatanswer, name='chatanswer'),


]