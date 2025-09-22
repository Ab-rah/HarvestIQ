from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.generic import TemplateView
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
import pandas as pd
import os
from .models import CSVUpload, ScrapedURL
from .serializers import CSVUploadSerializer, ScrapedURLSerializer
from .scraper import WebScraper
from .vector_db import VectorDatabase
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)


class HomeView(TemplateView):
    """Home page with upload and search functionality"""
    template_name = 'knowledge_base/home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['recent_uploads'] = CSVUpload.objects.order_by('-uploaded_at')[:5]
        return context


def upload_csv(request):
    """Handle CSV file upload - NO CELERY"""
    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']

        # Validate file extension
        if not csv_file.name.lower().endswith(('.csv', '.xlsx', '.xls')):
            messages.error(request, 'Please upload a CSV or Excel file.')
            return redirect('home')

        try:
            # Create CSVUpload record
            csv_upload = CSVUpload.objects.create(file=csv_file)

            # Read the file
            if csv_file.name.lower().endswith('.csv'):
                df = pd.read_csv(csv_file)
            else:
                df = pd.read_excel(csv_file)

            # Look for URL columns
            url_column = None
            for col in df.columns:
                if 'url' in col.lower() or 'link' in col.lower():
                    url_column = col
                    break

            if url_column is None:
                # Try first column if no URL column found
                url_column = df.columns[0]
                messages.warning(request, f'No URL column found. Using "{url_column}" column.')

            # Extract URLs and create ScrapedURL records
            urls = df[url_column].dropna().unique()

            scraped_urls = []
            for url in urls:
                if isinstance(url, str) and (url.startswith('http://') or url.startswith('https://')):
                    scraped_urls.append(
                        ScrapedURL(csv_upload=csv_upload, url=url)
                    )

            # Bulk create
            ScrapedURL.objects.bulk_create(scraped_urls, ignore_conflicts=True)

            # Update total URLs count
            csv_upload.total_urls = len(scraped_urls)
            csv_upload.save()

            # NO CELERY TASK - Just redirect to status page
            messages.success(request,
                             f'CSV uploaded successfully! Found {len(scraped_urls)} URLs. Click "Process URLs" to start scraping.')
            return redirect('upload_status', upload_id=csv_upload.id)

        except Exception as e:
            messages.error(request, f'Error processing file: {str(e)}')
            return redirect('home')

    return redirect('home')


def upload_status(request, upload_id):
    """Show upload processing status with Process button"""
    csv_upload = get_object_or_404(CSVUpload, id=upload_id)
    progress_percent = (csv_upload.processed_urls / csv_upload.total_urls * 100) if csv_upload.total_urls > 0 else 0

    context = {
        'csv_upload': csv_upload,
        'progress_percent': progress_percent,
        'can_process': csv_upload.scraped_urls.filter(scraping_status='pending').exists()
    }
    return render(request, 'knowledge_base/upload_status.html', context)


def process_urls(request, upload_id):
    """MANUAL PROCESSING - Process URLs synchronously"""
    csv_upload = get_object_or_404(CSVUpload, id=upload_id)

    if request.method == 'POST':
        try:
            # Get pending URLs
            pending_urls = csv_upload.scraped_urls.filter(scraping_status='pending')

            if not pending_urls.exists():
                messages.info(request, 'No pending URLs to process.')
                return redirect('upload_status', upload_id=upload_id)

            # Initialize scraper
            scraper = WebScraper()
            processed_count = 0

            # Process each URL
            for scraped_url in pending_urls:
                logger.info(f"Processing URL: {scraped_url.url}")

                # Update status to processing
                scraped_url.scraping_status = 'processing'
                scraped_url.save()

                try:
                    # Scrape the URL
                    result = scraper.scrape_url(scraped_url.url)

                    # Update the record with results
                    scraped_url.title = result['title'][:500] if result['title'] else ''
                    scraped_url.content = result['content'][:10000] if result['content'] else ''  # Limit content size
                    scraped_url.meta_description = result['meta_description'][:1000] if result[
                        'meta_description'] else ''
                    scraped_url.status_code = result['status_code']
                    scraped_url.error_message = result['error_message'][:1000] if result['error_message'] else ''
                    scraped_url.structured_data = result['structured_data']
                    scraped_url.scraped_at = timezone.now()

                    if result['status_code'] == 200 and result['content']:
                        scraped_url.scraping_status = 'success'
                        processed_count += 1
                        logger.info(f"Successfully processed: {scraped_url.url}")
                    else:
                        scraped_url.scraping_status = 'failed'
                        logger.warning(f"Failed to process: {scraped_url.url} - {result['error_message']}")

                except Exception as e:
                    scraped_url.scraping_status = 'failed'
                    scraped_url.error_message = str(e)[:1000]
                    logger.error(f"Error processing {scraped_url.url}: {e}")

                scraped_url.save()

                # Update progress counter
                csv_upload.processed_urls = csv_upload.scraped_urls.exclude(scraping_status='pending').count()
                csv_upload.save()

            # Mark as processed if all URLs are done
            if not csv_upload.scraped_urls.filter(scraping_status='pending').exists():
                csv_upload.processed = True
                csv_upload.save()

                # Add successful documents to vector database
                try:
                    add_to_vector_database(csv_upload.id)
                    messages.success(request,
                                     f'Processing complete! {processed_count} URLs processed successfully. Documents added to search index.')
                except Exception as e:
                    messages.warning(request,
                                     f'Processing complete! {processed_count} URLs processed, but vector database update failed: {str(e)}')
            else:
                messages.success(request,
                                 f'Processed {processed_count} URLs successfully. Some URLs may still be pending.')

        except Exception as e:
            messages.error(request, f'Processing error: {str(e)}')
            logger.error(f"Processing error for upload {upload_id}: {e}")

    return redirect('upload_status', upload_id=upload_id)


def add_to_vector_database(csv_upload_id):
    """Add successfully scraped documents to vector database"""
    csv_upload = CSVUpload.objects.get(id=csv_upload_id)
    successful_urls = csv_upload.scraped_urls.filter(
        scraping_status='success',
        vectorized=False
    )

    if not successful_urls.exists():
        return

    vector_db = VectorDatabase()

    documents = []
    metadata = []

    for scraped_url in successful_urls:
        # Combine title, content, and structured data for embedding
        doc_text = f"{scraped_url.title}\n{scraped_url.content}\n{scraped_url.meta_description}"

        # Add structured data to document text
        if scraped_url.structured_data:
            struct_text = []
            if 'person' in scraped_url.structured_data:
                person = scraped_url.structured_data['person']
                if 'names' in person:
                    struct_text.append(f"Names: {', '.join(person['names'])}")
                if 'titles' in person:
                    struct_text.append(f"Titles: {', '.join(person['titles'])}")
                if 'bios' in person:
                    struct_text.extend(person['bios'])

            if struct_text:
                doc_text += "\n" + "\n".join(struct_text)

        documents.append(doc_text)
        metadata.append({
            'id': scraped_url.id,
            'url': scraped_url.url,
            'title': scraped_url.title,
            'structured_data': scraped_url.structured_data,
            'csv_upload_id': csv_upload_id
        })

    # Add to vector database
    vector_db.add_documents(documents, metadata)

    # Mark as vectorized
    successful_urls.update(vectorized=True)

    logger.info(f"Vectorized {len(documents)} documents from CSV {csv_upload_id}")


# def search_knowledge_base(request):
#     """Search the knowledge base using vector similarity"""
#     query = request.GET.get('q', '')
#
#     if not query:
#         return render(request, 'knowledge_base/search_results.html', {
#             'query': query,
#             'results': [],
#             'error': 'Please enter a search query.'
#         })
#
#     try:
#         vector_db = VectorDatabase()
#         results = vector_db.search(query, top_k=20)
#
#         # Enhance results with database records
#         enhanced_results = []
#         for result in results:
#             try:
#                 scraped_url = ScrapedURL.objects.get(id=result['metadata']['id'])
#                 result['scraped_url'] = scraped_url
#                 result['similarity_percent'] = round(result['similarity_score'] * 100)
#                 enhanced_results.append(result)
#             except ScrapedURL.DoesNotExist:
#                 continue
#
#         return render(request, 'knowledge_base/search_results.html', {
#             'query': query,
#             'results': enhanced_results,
#             'total_found': len(enhanced_results)
#         })
#
#     except Exception as e:
#         return render(request, 'knowledge_base/search_results.html', {
#             'query': query,
#             'results': [],
#             'error': f'Search error: {str(e)}'
#         })
#
#     # API Views)

def search_knowledge_base(request):
    """Enhanced search with multiple modes and filters - Fixed template issues"""
    query = request.GET.get('q', '')
    search_mode = request.GET.get('mode', 'hybrid')  # hybrid, similarity, relevance
    min_similarity = float(request.GET.get('min_similarity', 0.3))

    # Define similarity thresholds for template
    similarity_thresholds = [0.2, 0.3, 0.4, 0.5]

    if not query:
        return render(request, 'knowledge_base/search_results.html', {
            'query': query,
            'results': [],
            'search_modes': ['hybrid', 'similarity', 'relevance'],
            'current_mode': search_mode,
            'min_similarity': min_similarity,
            'similarity_thresholds': similarity_thresholds,
            'error': 'Please enter a search query.'
        })

    try:
        vector_db = VectorDatabase()

        # Use advanced search for executive-related queries
        boost_terms = []
        if any(term in query.lower() for term in ['ceo', 'executive', 'director', 'manager', 'president']):
            boost_terms = ['ceo', 'executive', 'director', 'president', 'leadership', 'management']

        # Apply filters for person data if relevant
        filters = {}
        if any(term in query.lower() for term in ['person', 'people', 'executive', 'bio', 'profile']):
            filters['has_person_data'] = True

        filters['min_similarity'] = min_similarity

        # Use advanced search with filters and boosts
        if boost_terms or filters:
            results = vector_db.advanced_search(
                query=query,
                filters=filters,
                boost_terms=boost_terms,
                top_k=20
            )
        else:
            results = vector_db.search(
                query=query,
                top_k=20,
                similarity_threshold=min_similarity,
                search_mode=search_mode
            )

        # Enhance results with database records
        enhanced_results = []
        for result in results:
            try:
                scraped_url = ScrapedURL.objects.get(id=result['metadata']['id'])
                result['scraped_url'] = scraped_url
                enhanced_results.append(result)
            except ScrapedURL.DoesNotExist:
                continue

        return render(request, 'knowledge_base/search_results.html', {
            'query': query,
            'results': enhanced_results,
            'total_found': len(enhanced_results),
            'search_modes': ['hybrid', 'similarity', 'relevance'],
            'current_mode': search_mode,
            'min_similarity': min_similarity,
            'similarity_thresholds': similarity_thresholds,
            'stats': vector_db.get_stats()
        })

    except Exception as e:
        logger.error(f"Search error: {e}")
        return render(request, 'knowledge_base/search_results.html', {
            'query': query,
            'results': [],
            'search_modes': ['hybrid', 'similarity', 'relevance'],
            'current_mode': search_mode,
            'min_similarity': min_similarity,
            'similarity_thresholds': similarity_thresholds,
            'error': f'Search error: {str(e)}'
        })
class CSVUploadViewSet(viewsets.ReadOnlyModelViewSet):
    """API for CSV uploads"""
    queryset = CSVUpload.objects.all().order_by('-uploaded_at')
    serializer_class = CSVUploadSerializer

class ScrapedURLViewSet(viewsets.ReadOnlyModelViewSet):
    """API for scraped URLs"""
    queryset = ScrapedURL.objects.all().order_by('-scraped_at')

    serializer_class = ScrapedURLSerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        status = self.request.query_params.get('status')
        csv_id = self.request.query_params.get('csv_id')

        if status:
            queryset = queryset.filter(scraping_status=status)
        if csv_id:
            queryset = queryset.filter(csv_upload_id=csv_id)

        return queryset

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get scraping statistics"""
        total = self.get_queryset().count()
        success = self.get_queryset().filter(scraping_status='success').count()
        failed = self.get_queryset().filter(scraping_status='failed').count()
        pending = self.get_queryset().filter(scraping_status='pending').count()
        processing = self.get_queryset().filter(scraping_status='processing').count()

        return Response({
            'total': total,
            'success': success,
            'failed': failed,
            'pending': pending,
            'processing': processing,
            'success_rate': (success / total * 100) if total > 0 else 0
        })
