import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import Dict, Any
import logging


logger = logging.getLogger(__name__)


class WebScraper:
    """Web scraper with content extraction and person detail detection"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        print("WebScraper initialized with custom User-Agent")

    def scrape_url(self, url: str) -> Dict[str, Any]:
        self.current_url = url
        """Scrape content from a single URL"""
        result = {
            'url': url,
            'title': '',
            'content': '',
            'meta_description': '',
            'status_code': None,
            'error_message': '',
            'structured_data': {}
        }

        try:
            response = self.session.get(url, timeout=30)
            result['status_code'] = response.status_code

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract basic content
                result['title'] = self._extract_title(soup)
                result['content'] = self._extract_content(soup)
                result['meta_description'] = self._extract_meta_description(soup)

                # Extract structured data (person details, executive info)
                result['structured_data'] = self._extract_structured_data(soup, url)

            else:
                result['error_message'] = f"HTTP {response.status_code}"

        except requests.exceptions.RequestException as e:
            result['error_message'] = str(e)
            logger.error(f"Error scraping {url}: {e}")
        except Exception as e:
            result['error_message'] = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error scraping {url}: {e}")

        return result

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()

        # Try h1 as fallback
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()

        return ""

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Try to find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div',
                                                                              class_=re.compile(r'content|main|body'))

        if main_content:
            text = main_content.get_text()
        else:
            text = soup.get_text()

        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text[:10000]  # Limit content size

    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return meta_desc.get('content', '').strip()

        # Try Open Graph description
        og_desc = soup.find('meta', attrs={'property': 'og:description'})
        if og_desc:
            return og_desc.get('content', '').strip()

        return ""

    def _extract_person_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract person details (names, titles, bios, etc.)"""
        person_info = {}

        # Look for common person-related patterns
        name_patterns = [
            r'(?:CEO|CTO|CFO|President|Director|Manager|Executive)\s*[:-]\s*([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)(?:\s*,\s*(?:CEO|CTO|CFO|President|Director))',
        ]

        text = soup.get_text()

        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                person_info['names'] = list(set(matches))
                break

        # Look for job titles
        title_patterns = [
            r'(CEO|Chief Executive Officer|CTO|Chief Technology Officer|CFO|Chief Financial Officer)',
            r'(President|Vice President|VP|Director|Manager|Executive)',
        ]

        titles = []
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            titles.extend(matches)

        if titles:
            person_info['titles'] = list(set(titles))

        # Look for bio sections
        bio_sections = soup.find_all(['div', 'section', 'p'],
                                     class_=re.compile(r'bio|about|profile|description', re.I))
        if bio_sections:
            bios = []
            for section in bio_sections[:3]:  # Limit to first 3
                bio_text = section.get_text().strip()
                if len(bio_text) > 50:  # Only meaningful bios
                    bios.append(bio_text[:500])  # Limit bio length

            if bios:
                person_info['bios'] = bios

        return person_info

    def _extract_structured_data(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract structured data focusing on person details and executive information"""
        structured_data = {}

        # Extract person-related information
        person_info = self._extract_person_info(soup)
        if person_info:
            print(f"Person info found on {url}: {person_info}")
            structured_data['person'] = person_info

        # Extract company/organization info
        company_info = self._extract_company_info(soup)
        if company_info:
            print(f"Company info found on {url}: {company_info}")
            structured_data['company'] = company_info

        # Extract contact information
        contact_info = self._extract_contact_info(soup)
        if contact_info:
            print(f"Contact info found on {url}: {contact_info}")
            structured_data['contact'] = contact_info

        # Extract social media links
        social_links = self._extract_social_links(soup)
        if social_links:
            print(f"Social links found on {url}: {social_links}")
            structured_data['social'] = social_links

        return structured_data

    def _extract_company_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract company/organization information"""
        company_info = {}

        # Look for company name in title or headers
        title = soup.find('title')
        if title:
            title_text = title.get_text()
            # Common company patterns
            company_patterns = [
                r'([A-Z][a-zA-Z\s]+(?:Inc|LLC|Corp|Company|Corporation|Ltd))',
                r'([A-Z][a-zA-Z\s]+)\s*[-|–]\s*(?:Home|About|Company)',
            ]

            for pattern in company_patterns:
                match = re.search(pattern, title_text)
                if match:
                    company_info['name'] = match.group(1).strip()
                    break

        return company_info

    def _extract_contact_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract contact information"""
        contact_info = {}
        text = soup.get_text()

        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['emails'] = list(set(emails))

        # Extract phone numbers
        phone_pattern = r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phones'] = list(set(phones))

        return contact_info

    def _extract_contact_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract contact information"""
        contact_info = {}
        text = soup.get_text()

        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['emails'] = list(set(emails))

        # Extract phone numbers
        phone_pattern = r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phones'] = list(set(phones))

        return contact_info

    def _extract_social_links(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract social media links"""
        social_info = {}
        social_patterns = {
            'linkedin': r'linkedin\.com/(?:in/|company/)[^/\s]+',
            'twitter': r'twitter\.com/[^/\s]+',
            'facebook': r'facebook\.com/[^/\s]+',
            'instagram': r'instagram\.com/[^/\s]+',
        }

        # Get all links
        links = soup.find_all('a', href=True)

        for platform, pattern in social_patterns.items():
            platform_links = []
            for link in links:
                href = link.get('href', '')
                if re.search(pattern, href, re.IGNORECASE):
                    platform_links.append(href)

            if platform_links:
                social_info[platform] = list(set(platform_links))

        return social_info