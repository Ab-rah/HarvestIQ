from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# API Router
router = DefaultRouter()
router.register(r'uploads', views.CSVUploadViewSet)
router.register(r'urls', views.ScrapedURLViewSet)

urlpatterns = [
    # Web UI
    path('', views.HomeView.as_view(), name='home'),
    path('upload/', views.upload_csv, name='upload_csv'),
    path('upload/<int:upload_id>/status/', views.upload_status, name='upload_status'),
    path('upload/<int:upload_id>/process/', views.process_urls, name='process_urls'),  # NEW!
    path('search/', views.search_knowledge_base, name='search'),

    # API
    path('api/', include(router.urls)),
]
