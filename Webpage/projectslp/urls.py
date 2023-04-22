from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path("home", views.index, name="home"),
    path("summary", views.summary_page, name="summary"),
    path("translate", views.translater, name="translate"),
    path("ts", views.ts, name="ts"),
    path("hsum", views.hsum, name="hsum"),
    path("output", views.output, name='output'),
]