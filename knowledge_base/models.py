from django.db import models
from django.utils import timezone
import json


class CSVUpload(models.Model):
    """Track CSV file uploads"""
    file = models.FileField(upload_to='csv_uploads/')
    uploaded_at = models.DateTimeField(default=timezone.now)
    processed = models.BooleanField(default=False)
    total_urls = models.IntegerField(default=0)
    processed_urls = models.IntegerField(default=0)

    def __str__(self):
        return f"CSV Upload - {self.file.name} ({self.uploaded_at})"


class ScrapedURL(models.Model):
    """Store scraped content from URLs"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('success', 'Success'),
        ('failed', 'Failed'),
        ('processing', 'Processing')
    ]

    csv_upload = models.ForeignKey(CSVUpload, on_delete=models.CASCADE, related_name='scraped_urls')
    url = models.URLField(max_length=2000)
    title = models.CharField(max_length=500, blank=True)
    content = models.TextField(blank=True)
    meta_description = models.TextField(blank=True)
    status_code = models.IntegerField(null=True)
    scraping_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    scraped_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    # Extracted structured data (for person details, executive info, etc.)
    structured_data = models.JSONField(default=dict, blank=True)

    # Vector DB integration
    vectorized = models.BooleanField(default=False)
    vector_id = models.CharField(max_length=100, blank=True)

    class Meta:
        unique_together = ['csv_upload', 'url']

    def __str__(self):
        return f"{self.url} - {self.scraping_status}"
