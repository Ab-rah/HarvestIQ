from rest_framework import serializers
from .models import CSVUpload, ScrapedURL


class ScrapedURLSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScrapedURL
        fields = [
            'id', 'url', 'title', 'content', 'meta_description',
            'status_code', 'scraping_status', 'scraped_at',
            'error_message', 'structured_data', 'vectorized'
        ]
        read_only_fields = ['id', 'scraped_at', 'vectorized']


class CSVUploadSerializer(serializers.ModelSerializer):
    scraped_urls = ScrapedURLSerializer(many=True, read_only=True)

    class Meta:
        model = CSVUpload
        fields = [
            'id', 'file', 'uploaded_at', 'processed',
            'total_urls', 'processed_urls', 'scraped_urls'
        ]
        read_only_fields = ['id', 'uploaded_at', 'processed', 'total_urls', 'processed_urls']