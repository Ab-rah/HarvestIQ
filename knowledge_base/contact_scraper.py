import re
from typing import Dict, Any, Set, List, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging
import requests

logger = logging.getLogger(__name__)


class DynamicPersonExtractor:
    """Dynamic extraction system for executives and company information"""

    def __init__(self):
        # Dynamic patterns that work for any name/company
        self.executive_title_patterns = [
            r'(?:chief\s+executive\s+officer?|ceo)',
            r'(?:chief\s+technology\s+officer?|cto)',
            r'(?:chief\s+financial\s+officer?|cfo)',
            r'(?:chief\s+operating\s+officer?|coo)',
            r'(?:chief\s+marketing\s+officer?|cmo)',
            r'(?:chief\s+information\s+officer?|cio)',
            r'(?:president)',
            r'(?:vice\s+president|vp)',
            r'(?:executive\s+director)',
            r'(?:managing\s+director)',
            r'(?:founder|co-?founder)',
            r'(?:chairman|chairwoman|chair)',
            r'(?:director)',
            r'(?:manager)',
        ]

        # Common word patterns that appear in names vs titles
        self.name_indicators = [
            r'[A-Z][a-z]+',  # Capitalized words
            r'[A-Z]\.',  # Initials like "J."
            r'[A-Z][a-z]+\-[A-Z][a-z]+',  # Hyphenated names
            r'[A-Z]\'[A-Z][a-z]+',  # Names with apostrophes
        ]

    def extract_person_info(self, soup: BeautifulSoup, url: str = "") -> Dict[str, Any]:
        """Dynamically extract person information without hardcoding"""
        text = soup.get_text()
        person_info = {}

        # Extract company information dynamically first
        company_info = self._extract_company_dynamically(soup, url, text)
        if company_info:
            person_info['company'] = company_info

        # Extract executive names and titles dynamically
        executives = self._extract_executives_dynamically(text)
        if executives:
            person_info['names'] = [exec['name'] for exec in executives]
            person_info['titles'] = [exec['title'] for exec in executives]
            person_info['name_title_pairs'] = [(exec['name'], exec['title']) for exec in executives]

        # Extract biographical content dynamically
        bios = self._extract_bios_dynamically(soup)
        if bios:
            person_info['bios'] = bios

        # Extract contact information dynamically
        contacts = self._extract_contacts_dynamically(text)
        if contacts:
            person_info.update(contacts)

        return person_info

    def _extract_company_dynamically(self, soup: BeautifulSoup, url: str, text: str) -> str:
        """Dynamically extract company name from various sources"""
        company_candidates = set()

        # Method 1: Extract from URL domain
        if url:
            domain = urlparse(url).netloc.lower()
            domain_parts = domain.replace('www.', '').split('.')
            if domain_parts:
                company_name = domain_parts[0].title()
                if len(company_name) > 2:
                    company_candidates.add(company_name)

        # Method 2: Extract from page title
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text()
            # Look for "Company - Page" or "Page | Company" patterns
            title_patterns = [
                r'^([^|\-–]+)(?:\s*[\|\-–]\s*.+)?$',  # "Apple - Leadership"
                r'^.+[\|\-–]\s*([^|\-–]+)$',  # "Leadership | Apple"
            ]

            for pattern in title_patterns:
                match = re.search(pattern, title.strip())
                if match:
                    candidate = match.group(1).strip()
                    if self._is_valid_company_name(candidate):
                        company_candidates.add(candidate)

        # Method 3: Look for "Company Name" in structured content
        # Find copyright notices, headers, footers
        structured_elements = soup.find_all(['footer', 'header', 'nav'])
        for element in structured_elements:
            element_text = element.get_text()
            # Look for copyright patterns: "© 2024 Company Name"
            copyright_match = re.search(r'©\s*\d{4}\s+([^.\n,]+)', element_text)
            if copyright_match:
                candidate = copyright_match.group(1).strip()
                if self._is_valid_company_name(candidate):
                    company_candidates.add(candidate)

        # Method 4: Extract from executive context
        # Look for "CEO of Company" patterns
        ceo_company_patterns = [
            r'(?:ceo|chief\s+executive\s+officer?)\s+(?:of|at)\s+([A-Z][^.\n,]{2,30})',
            r'([A-Z][^.\n,]{2,30})\s+(?:ceo|chief\s+executive\s+officer?)',
            r'president\s+(?:of|at)\s+([A-Z][^.\n,]{2,30})',
            r'([A-Z][^.\n,]{2,30})\s+president',
        ]

        for pattern in ceo_company_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                candidate = match.strip()
                if self._is_valid_company_name(candidate):
                    company_candidates.add(candidate)

        # Method 5: Look for "About Company" or "Company Leadership" patterns
        about_patterns = [
            r'about\s+([A-Z][^.\n,]{2,30})',
            r'([A-Z][^.\n,]{2,30})\s+leadership',
            r'([A-Z][^.\n,]{2,30})\s+team',
            r'([A-Z][^.\n,]{2,30})\s+executives',
        ]

        for pattern in about_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                candidate = match.strip()
                if self._is_valid_company_name(candidate):
                    company_candidates.add(candidate)

        # Return the best candidate (shortest, most likely to be company name)
        if company_candidates:
            return min(company_candidates, key=len)

        return None

    def _extract_executives_dynamically(self, text: str) -> List[Dict[str, str]]:
        """Dynamically extract executive names and titles without hardcoding"""
        executives = []

        # Create comprehensive patterns that work for any name
        extraction_patterns = [
            # Pattern 1: "Title: Name" or "Title Name"
            r'(?:^|\n|\.|,)\s*((?:' + '|'.join(
                self.executive_title_patterns) + r'))\s*[:\-–]?\s*([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-\.]*){1,3})',

            # Pattern 2: "Name, Title" or "Name - Title"
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-\.]*){1,3})\s*[,\-–]\s*((?:' + '|'.join(
                self.executive_title_patterns) + r'))',

            # Pattern 3: "Name is the Title" or "Name serves as Title"
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-\.]*){1,3})\s+(?:is|serves?\s+as|works?\s+as)\s+(?:the\s+)?((?:' + '|'.join(
                self.executive_title_patterns) + r'))',

            # Pattern 4: "Title of Company, Name" (when company is mentioned)
            r'((?:' + '|'.join(
                self.executive_title_patterns) + r'))\s+(?:of|at)\s+[^,\n]*[,\s]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-\.]*){1,3})',

            # Pattern 5: Leadership listings "• Name - Title" or "Name (Title)"
            r'[•\*\-]?\s*([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-\.]*){1,3})\s*[\-–\(]\s*((?:' + '|'.join(
                self.executive_title_patterns) + r'))',

            # Pattern 6: Bio introductions "Name, Title at Company"
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-\.]*){1,3}),\s*((?:' + '|'.join(
                self.executive_title_patterns) + r'))\s+(?:at|of)',
        ]

        seen_combinations = set()

        for pattern in extraction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)

            for match in matches:
                if len(match) == 2:
                    # Determine which is name and which is title
                    part1, part2 = match

                    if self._looks_like_executive_title(part1):
                        title, name = part1, part2
                    else:
                        name, title = part1, part2

                    # Clean and validate
                    name = self._clean_name(name)
                    title = self._clean_title(title)

                    if (self._is_valid_person_name(name) and
                            self._looks_like_executive_title(title) and
                            (name, title) not in seen_combinations):
                        executives.append({
                            'name': name,
                            'title': title,
                            'confidence': self._calculate_confidence(name, title)
                        })
                        seen_combinations.add((name, title))

        # Sort by confidence and return top results
        executives.sort(key=lambda x: x['confidence'], reverse=True)
        return executives[:10]  # Limit to top 10 most confident matches

    def _extract_bios_dynamically(self, soup: BeautifulSoup) -> List[str]:
        """Dynamically extract biographical information"""
        bio_selectors = [
            # Generic bio containers
            {'class': re.compile(r'bio|biography|profile|about|description|summary', re.I)},
            {'id': re.compile(r'bio|biography|profile|about|description', re.I)},

            # Executive-specific containers
            {'class': re.compile(r'executive|leadership|team|staff', re.I)},
            {'id': re.compile(r'executive|leadership|team', re.I)},

            # Schema.org structured data
            {'itemtype': re.compile(r'schema\.org/Person', re.I)},
            {'itemscope': True, 'itemtype': True},
        ]

        bio_texts = []

        for selector in bio_selectors:
            elements = soup.find_all(['div', 'section', 'article', 'p'], attrs=selector)

            for element in elements:
                text = element.get_text().strip()

                # Filter for substantial biographical content
                if (len(text) > 100 and
                        len(text) < 2000 and
                        self._looks_like_biographical_content(text)):
                    # Clean the text
                    cleaned_text = re.sub(r'\s+', ' ', text)
                    bio_texts.append(cleaned_text)

        # Remove duplicates and return unique bios
        unique_bios = []
        for bio in bio_texts:
            if not any(bio in existing or existing in bio for existing in unique_bios):
                unique_bios.append(bio)

        return unique_bios[:3]  # Limit to top 3 bios

    def _extract_contacts_dynamically(self, text: str) -> Dict[str, List[str]]:
        """Dynamically extract contact information"""
        contacts = {}

        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contacts['emails'] = list(set(emails))  # Remove duplicates

        # Phone extraction (various formats)
        phone_patterns = [
            r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',  # US format
            r'\+?([0-9]{1,4})[-.\s]?([0-9]{2,4})[-.\s]?([0-9]{2,4})[-.\s]?([0-9]{2,4})',  # International
        ]

        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    phone = ''.join(match)
                    if len(phone) >= 10:  # Valid phone length
                        phones.append('-'.join(match))

        if phones:
            contacts['phones'] = list(set(phones))

        return contacts

    def _is_valid_company_name(self, candidate: str) -> bool:
        """Check if extracted text looks like a valid company name"""
        if not candidate or len(candidate) < 2 or len(candidate) > 50:
            return False

        # Remove common false positives
        false_positives = {
            'home', 'about', 'contact', 'news', 'blog', 'careers', 'products',
            'services', 'team', 'leadership', 'executives', 'board', 'staff',
            'privacy', 'terms', 'legal', 'copyright', 'rights', 'reserved',
            'website', 'site', 'page', 'content', 'information', 'data',
            'the', 'and', 'or', 'of', 'at', 'in', 'on', 'for', 'with'
        }

        if candidate.lower() in false_positives:
            return False

        # Must start with capital letter
        if not candidate[0].isupper():
            return False

        # Should contain mostly letters and common business words
        if not re.match(r'^[A-Za-z0-9\s&.,\-\']+$', candidate):
            return False

        return True

    def _is_valid_person_name(self, name: str) -> bool:
        """Dynamic validation for person names"""
        if not name or len(name) < 2 or len(name) > 50:
            return False

        words = name.split()

        # Must be 2-4 words (first name, last name, optionally middle name/initial)
        if not 2 <= len(words) <= 4:
            return False

        # Each word should be proper case or initial
        for word in words:
            if not word:
                return False

            # Must start with capital
            if not word[0].isupper():
                return False

            # Should be alphabetic with common name characters
            if not re.match(r'^[A-Za-z\'\-\.]+$', word):
                return False

        # Avoid common false positives
        false_names = {
            'About Us', 'Contact Us', 'Our Team', 'Leadership Team', 'Executive Team',
            'Board Members', 'Staff Members', 'Team Members', 'Press Release',
            'News Update', 'Latest News', 'Chief Executive', 'Executive Officer',
            'Vice President', 'Senior Director', 'General Manager', 'Project Manager'
        }

        if name in false_names:
            return False

        return True

    def _clean_name(self, name: str) -> str:
        """Clean extracted name"""
        # Remove extra whitespace and punctuation
        name = re.sub(r'[^\w\s\'\-\.]', ' ', name).strip()
        name = ' '.join(name.split())

        # Capitalize properly
        words = []
        for word in name.split():
            if len(word) == 1 or (len(word) == 2 and word.endswith('.')):
                words.append(word.upper())  # Initials
            else:
                words.append(word.capitalize())

        return ' '.join(words)

    def _clean_title(self, title: str) -> str:
        """Clean extracted title"""
        # Remove extra whitespace and normalize
        title = re.sub(r'\s+', ' ', title.strip())

        # Standardize common titles
        title_mappings = {
            r'chief\s+executive\s+officer?': 'CEO',
            r'chief\s+technology\s+officer?': 'CTO',
            r'chief\s+financial\s+officer?': 'CFO',
            r'chief\s+operating\s+officer?': 'COO',
            r'vice\s+president': 'Vice President',
            r'v\.?p\.?': 'VP',
        }

        title_lower = title.lower()
        for pattern, replacement in title_mappings.items():
            if re.search(pattern, title_lower):
                return replacement

        # Title case for other titles
        return title.title()

    def _looks_like_executive_title(self, text: str) -> bool:
        """Check if text looks like an executive title"""
        text_lower = text.lower()

        for pattern in self.executive_title_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def _looks_like_biographical_content(self, text: str) -> bool:
        """Check if text looks like biographical content"""
        bio_indicators = [
            r'\b(?:born|graduated|studied|worked|joined|founded|led|managed|responsible)\b',
            r'\b(?:experience|background|education|career|achievements|accomplishments)\b',
            r'\b(?:degree|university|college|school|mba|phd|bachelor|master)\b',
            r'\b(?:years?|decades?)\s+(?:of\s+)?(?:experience|work|leadership)\b',
            r'\b(?:prior|previous|before|after|currently|now|today)\b',
        ]

        text_lower = text.lower()
        matches = sum(1 for pattern in bio_indicators if re.search(pattern, text_lower))

        # Require at least 2 biographical indicators
        return matches >= 2

    def _calculate_confidence(self, name: str, title: str) -> float:
        """Calculate confidence score for name-title pairing"""
        confidence = 0.5  # Base confidence

        # Higher confidence for CEO/President titles
        if re.search(r'ceo|chief executive|president', title.lower()):
            confidence += 0.3

        # Higher confidence for proper name structure
        words = name.split()
        if len(words) == 2:  # First Last
            confidence += 0.2
        elif len(words) == 3:  # First Middle Last or First M. Last
            confidence += 0.1

        # Higher confidence for clean extraction (no punctuation issues)
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', name):
            confidence += 0.1

        return min(confidence, 1.0)


# Integration with existing WebScraper class
def _extract_person_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
    """Dynamic person extraction (replace in WebScraper class)"""
    extractor = DynamicPersonExtractor()
    return extractor.extract_person_info(soup, getattr(self, 'current_url', ''))


def _extract_company_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
    """Dynamic company extraction (replace in WebScraper class)"""
    extractor = DynamicPersonExtractor()
    company_name = extractor._extract_company_dynamically(soup, getattr(self, 'current_url', ''), soup.get_text())

    if company_name:
        return {'name': company_name}
    return {}


# Update WebScraper.scrape_url method to pass URL context
def scrape_url(self, url: str) -> Dict[str, Any]:
    """Enhanced scrape_url method that provides URL context"""
    self.current_url = url  # Store URL for dynamic extraction

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

            # Extract structured data using dynamic system
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