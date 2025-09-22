from celery import shared_task
from django.utils import timezone
from .models import CSVUpload, ScrapedURL
from .scraper import WebScraper
from .vector_db import VectorDatabase
import logging

logger = logging.getLogger(__name__)


@shared_task
def process_csv_urls(csv_upload_id: int):
    """Background task to scrape URLs from CSV upload"""
    try:
        csv_upload = CSVUpload.objects.get(id=csv_upload_id)
        print(f"Processing CSVUpload id={csv_upload_id}")
        scraper = WebScraper()

        urls_to_process = csv_upload.scraped_urls.filter(scraping_status='pending')

        for scraped_url in urls_to_process:
            print(f"Scraping URL: {scraped_url.url}")
            # Update status to processing
            scraped_url.scraping_status = 'processing'
            scraped_url.save()

            # Scrape the URL
            result = scraper.scrape_url(scraped_url.url)

            # Update the record with results
            scraped_url.title = result['title']
            scraped_url.content = result['content']
            scraped_url.meta_description = result['meta_description']
            scraped_url.status_code = result['status_code']
            scraped_url.error_message = result['error_message']
            scraped_url.structured_data = result['structured_data']
            scraped_url.scraped_at = timezone.now()

            if result['status_code'] == 200 and result['content']:
                scraped_url.scraping_status = 'success'
            else:
                scraped_url.scraping_status = 'failed'

            scraped_url.save()

            # Update progress
            csv_upload.processed_urls += 1
            csv_upload.save()

            logger.info(f"Processed URL: {scraped_url.url} - Status: {scraped_url.scraping_status}")

        # Mark CSV as processed
        csv_upload.processed = True
        csv_upload.save()

        # Add successful documents to vector database
        vectorize_documents.delay(csv_upload_id)

    except Exception as e:
        logger.error(f"Error processing CSV {csv_upload_id}: {e}")


@shared_task
def vectorize_documents(csv_upload_id: int):
    """Add successfully scraped documents to vector database"""
    try:
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

    except Exception as e:
        logger.error(f"Error vectorizing documents for CSV {csv_upload_id}: {e}")