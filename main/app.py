import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import textstat
import datetime


# =============================================================================
# NLTK DATA SETUP AND INITIALIZATION
# =============================================================================

def setup_nltk_data():
    """Download and verify required NLTK data with comprehensive error handling"""
    required_data = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords')
    ]

    print("🔧 Setting up NLTK data...")

    for data_path, download_name in required_data:
        try:
            nltk.data.find(data_path)
            print(f"✅ {download_name} already available")
        except LookupError:
            print(f"📦 Downloading {download_name}...")
            try:
                nltk.download(download_name, quiet=True)
                print(f"✅ {download_name} downloaded successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not download {download_name}: {e}")

    print("✅ NLTK setup complete!\n")


# Initialize NLTK data
setup_nltk_data()

# =============================================================================
# ENVIRONMENT SETUP AND CONFIGURATION
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# Environment variable validation
required_env_vars = ['MONGO_URI', 'MONGO_DB', 'MONGO_COLLECTION', 'PERPLEXITY_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
    print("Please ensure your .env file contains all required variables.")
    exit(1)

# Load configuration
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

print("🔧 Initializing system components...")

# =============================================================================
# DATABASE AND MODEL INITIALIZATION
# =============================================================================

try:
    # Connect to MongoDB with error handling
    print("📡 Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    collection = client[MONGO_DB][MONGO_COLLECTION]

    # Test connection
    client.admin.command('ping')
    print("✅ MongoDB connection successful")

except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit(1)

try:
    # Load embedding model
    print("🤖 Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Model loaded successfully")

except Exception as e:
    print(f"❌ Failed to load embedding model: {e}")
    exit(1)


# =============================================================================
# RAG EVALUATION CLASS WITH COMPREHENSIVE METRICS
# =============================================================================

class RAGEvaluator:
    """
    Comprehensive RAG evaluation system that measures response quality
    across multiple dimensions including semantic similarity, coherence,
    factual grounding, and context utilization.
    """

    def __init__(self, embedding_model):
        """
        Initialize the evaluator with embedding model and stopwords

        Args:
            embedding_model: SentenceTransformer model for semantic similarity
        """
        self.model = embedding_model

        # Initialize stopwords with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
            print("✅ Stopwords loaded successfully")
        except LookupError:
            print("⚠️ Stopwords not found, attempting download...")
            try:
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('english'))
                print("✅ Stopwords downloaded and loaded")
            except Exception as e:
                print(f"❌ Could not load stopwords: {e}")
                self.stop_words = set()  # Use empty set as fallback

    def safe_tokenize(self, text, method='word'):
        """
        Safe tokenization with fallback methods

        Args:
            text (str): Text to tokenize
            method (str): 'word' or 'sentence' tokenization

        Returns:
            list: Tokenized text
        """
        try:
            if method == 'word':
                return word_tokenize(text.lower())
            elif method == 'sentence':
                return sent_tokenize(text)
        except (LookupError, Exception):
            # Fallback to simple splitting
            if method == 'word':
                return text.lower().split()
            elif method == 'sentence':
                # Simple sentence splitting on periods, exclamation, question marks
                return re.split(r'[.!?]+', text)
        return []

    def calculate_semantic_similarity(self, query, response):
        """
        Calculate semantic similarity between query and response using embeddings

        Args:
            query (str): Original user query
            response (str): Generated response

        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            query_embedding = self.model.encode([query])
            response_embedding = self.model.encode([response])
            similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"⚠️ Error calculating semantic similarity: {e}")
            return 0.0

    def calculate_context_utilization(self, response, context_docs):
        """
        Measure how effectively the response utilizes provided context documents

        Args:
            response (str): Generated response
            context_docs (list): List of context documents

        Returns:
            float: Context utilization score between 0 and 1
        """
        if not context_docs:
            return 0.0

        try:
            response_embedding = self.model.encode([response])
            context_embeddings = self.model.encode(context_docs)
            similarities = cosine_similarity(response_embedding, context_embeddings)[0]
            return float(np.mean(similarities))
        except Exception as e:
            print(f"⚠️ Error calculating context utilization: {e}")
            return 0.0

    def calculate_information_density(self, text):
        """
        Calculate information density based on unique concepts and technical terminology

        Args:
            text (str): Text to analyze

        Returns:
            float: Information density score between 0 and 1
        """
        words = self.safe_tokenize(text, 'word')
        words = [w for w in words if w.isalpha() and w not in self.stop_words]

        if not words:
            return 0.0

        # Calculate unique word ratio
        unique_ratio = len(set(words)) / len(words)

        # Identify technical terms (words longer than 6 characters)
        technical_terms = [w for w in words if len(w) > 6]
        technical_ratio = len(technical_terms) / len(words) if words else 0

        # Combine metrics for overall density
        return (unique_ratio + technical_ratio) / 2

    def calculate_coherence_score(self, text):
        """
        Assess text coherence based on inter-sentence semantic relationships

        Args:
            text (str): Text to analyze

        Returns:
            float: Coherence score between 0 and 1
        """
        sentences = self.safe_tokenize(text, 'sentence')
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5  # Neutral score for single sentence

        try:
            # Calculate semantic similarity between adjacent sentences
            sentence_embeddings = self.model.encode(sentences)
            similarities = []

            for i in range(len(sentences) - 1):
                sim = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]
                similarities.append(sim)

            # Optimal coherence: moderate inter-sentence similarity (0.3-0.7 range)
            avg_similarity = np.mean(similarities)
            coherence_score = 1 - abs(avg_similarity - 0.5) / 0.5
            return max(0, min(1, float(coherence_score)))

        except Exception as e:
            print(f"⚠️ Error calculating coherence: {e}")
            return 0.5

    def calculate_specificity_score(self, text):
        """
        Measure response specificity based on concrete details and factual information

        Args:
            text (str): Text to analyze

        Returns:
            float: Specificity score between 0 and 1
        """
        # Patterns for specific information indicators
        specific_indicators = [
            r'\b\d+\b',  # Numbers
            r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',  # Proper nouns
            r'\b\d{4}\b',  # Years
            r'\$\d+',  # Dollar amounts
            r'\d+%',  # Percentages
            r'\b\w+\.com\b',  # URLs
            r'\b[A-Z]{2,}\b'  # Acronyms
        ]

        total_indicators = 0
        for pattern in specific_indicators:
            total_indicators += len(re.findall(pattern, text))

        words = len(self.safe_tokenize(text, 'word'))
        specificity = total_indicators / words if words > 0 else 0

        # Normalize to 0-1 scale with reasonable scaling factor
        return min(specificity * 10, 1.0)

    def calculate_readability_score(self, text):
        """
        Assess text readability using Flesch Reading Ease score

        Args:
            text (str): Text to analyze

        Returns:
            float: Readability score between 0 and 1
        """
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            # Convert Flesch score (0-100) to 0-1 scale
            readability = max(0, min(100, flesch_score)) / 100
            return readability
        except Exception as e:
            print(f"⚠️ Error calculating readability: {e}")
            return 0.5  # Neutral default

    def calculate_comprehensiveness(self, response, query):
        """
        Evaluate how comprehensively the response addresses the query

        Args:
            response (str): Generated response
            query (str): Original query

        Returns:
            float: Comprehensiveness score between 0 and 1
        """
        query_words = set(self.safe_tokenize(query, 'word'))
        response_words = set(self.safe_tokenize(response, 'word'))

        query_words = {w for w in query_words if w.isalpha() and w not in self.stop_words}
        response_words = {w for w in response_words if w.isalpha() and w not in self.stop_words}

        if not query_words:
            return 0.0

        # Calculate query concept coverage
        coverage = len(query_words.intersection(response_words)) / len(query_words)

        # Factor in response length (longer responses can be more comprehensive)
        length_factor = min(len(response.split()) / 100, 1.0)

        return (coverage + length_factor) / 2

    def calculate_factual_grounding(self, response, context_docs):
        """
        Assess how well the response is grounded in provided factual context

        Args:
            response (str): Generated response
            context_docs (list): Context documents

        Returns:
            float: Factual grounding score between 0 and 1
        """
        if not context_docs:
            return 0.0

        response_words = set(self.safe_tokenize(response, 'word'))
        response_words = {w for w in response_words if w.isalpha() and w not in self.stop_words}

        # Build vocabulary from all context documents
        context_words = set()
        for doc in context_docs:
            doc_words = set(self.safe_tokenize(doc, 'word'))
            doc_words = {w for w in doc_words if w.isalpha() and w not in self.stop_words}
            context_words.update(doc_words)

        if not response_words:
            return 0.0

        # Calculate vocabulary overlap between response and context
        overlap = len(response_words.intersection(context_words))
        grounding_score = overlap / len(response_words)

        return grounding_score

    def calculate_citation_accuracy(self, response, context_docs):
        """
        Measure citation accuracy and source attribution quality

        Args:
            response (str): Generated response
            context_docs (list): Context documents

        Returns:
            float: Citation accuracy score between 0 and 1
        """
        # Patterns for detecting citations and references
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\((\d+)\)',  # (1), (2), etc.
            r'according to',  # Attribution phrases
            r'as stated in',
            r'research shows',
            r'studies indicate'
        ]

        citations_found = 0
        for pattern in citation_patterns:
            citations_found += len(re.findall(pattern, response, re.IGNORECASE))

        words = len(response.split())
        citation_density = citations_found / max(words, 1) * 100

        # Normalize to reasonable scale
        return min(citation_density, 1.0)

    def calculate_hallucination_score(self, response, context_docs):
        """
        Detect potential hallucinations by verifying factual claims against context

        Args:
            response (str): Generated response
            context_docs (list): Context documents

        Returns:
            float: Hallucination prevention score (1.0 = no hallucinations)
        """
        if not context_docs:
            return 0.5  # Neutral score without context

        # Extract factual claims (numbers, dates, specific terms)
        factual_patterns = [
            r'\b\d+%\b',  # Percentages
            r'\b\d{4}\b',  # Years
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',  # Currency amounts
            r'\b[A-Z][a-z]+ \d+\b',  # "March 2023", "Version 3"
            r'\b\d+\.\d+\b'  # Version numbers, decimals
        ]

        response_facts = []
        for pattern in factual_patterns:
            response_facts.extend(re.findall(pattern, response))

        if not response_facts:
            return 0.7  # Slightly positive if no specific claims made

        # Verify facts against context documents
        context_text = ' '.join(context_docs).lower()
        verified_facts = sum(1 for fact in response_facts if fact.lower() in context_text)

        return verified_facts / len(response_facts) if response_facts else 0.7

    def calculate_temporal_relevance(self, response, query, context_docs):
        """
        Assess temporal relevance and currency of information

        Args:
            response (str): Generated response
            query (str): Original query
            context_docs (list): Context documents

        Returns:
            float: Temporal relevance score between 0 and 1
        """
        current_year = datetime.datetime.now().year

        # Extract years from response
        years_in_response = [int(year) for year in re.findall(r'\b(20\d{2})\b', response)]

        if not years_in_response:
            return 0.5  # Neutral if no temporal information

        # Calculate recency score (prefer recent information)
        avg_year = np.mean(years_in_response)
        years_old = current_year - avg_year

        # Score decays over 10 years, with recent info scoring higher
        recency_score = max(0, 1 - years_old / 10)

        return min(recency_score, 1.0)

    def calculate_diversity_score(self, response, context_docs):
        """
        Measure how well response integrates diverse information sources

        Args:
            response (str): Generated response
            context_docs (list): Context documents

        Returns:
            float: Source diversity score between 0 and 1
        """
        if not context_docs or len(context_docs) < 2:
            return 0.0

        try:
            response_embedding = self.model.encode([response])
            doc_embeddings = self.model.encode(context_docs)

            # Calculate similarity to each document
            similarities = cosine_similarity(response_embedding, doc_embeddings)[0]

            # Good diversity = balanced similarity across sources (low standard deviation)
            diversity_score = 1 - np.std(similarities)
            return max(0, min(1, diversity_score))

        except Exception as e:
            print(f"⚠️ Error calculating diversity: {e}")
            return 0.0

    def calculate_technical_depth(self, response, query):
        """
        Assess technical sophistication and domain expertise level

        Args:
            response (str): Generated response
            query (str): Original query

        Returns:
            float: Technical depth score between 0 and 1
        """
        # Technical sophistication indicators
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms (API, HTTP, etc.)
            r'\b\w+ly\b',  # Technical adverbs
            r'\b\w{10,}\b',  # Long technical terms
            r'\b\d+\.\d+\b',  # Version numbers, measurements
            r'\b\w+-\w+\b'  # Hyphenated technical terms
        ]

        technical_terms = 0
        for pattern in technical_patterns:
            technical_terms += len(re.findall(pattern, response))

        # Calculate average word length (technical writing tends to use longer words)
        words = [word for word in response.split() if word.isalpha()]
        avg_word_length = np.mean([len(word) for word in words]) if words else 0

        # Combine indicators
        words_count = len(words)
        technical_density = technical_terms / words_count if words_count else 0
        length_complexity = min(avg_word_length / 8, 1.0)  # Normalize around 8 characters

        # Weight technical terms more heavily
        return min((technical_density * 5 + length_complexity) / 2, 1.0)

    def calculate_logical_consistency(self, response):
        """
        Detect logical contradictions and assess internal consistency

        Args:
            response (str): Generated response

        Returns:
            float: Logical consistency score between 0 and 1
        """
        sentences = self.safe_tokenize(response, 'sentence')
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 1.0  # Perfect consistency for single sentence

        # Identify contradiction and negation indicators
        contradiction_words = ['however', 'but', 'although', 'despite', 'whereas', 'contradicts', 'opposes']
        negation_words = ['not', 'no', 'never', 'neither', 'none', 'without']

        contradiction_score = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            contradictions = sum(1 for word in contradiction_words if word in sentence_lower)
            negations = sum(1 for word in negation_words if word in sentence_lower)
            contradiction_score += (contradictions + negations * 0.5)

        # Normalize - some contradictions are natural in complex discussions
        normalized_score = 1 - min(contradiction_score / len(sentences), 0.5)
        return max(0, normalized_score)

    def calculate_completeness_score(self, response, query):
        """
        Evaluate how completely the response addresses query requirements

        Args:
            response (str): Generated response
            query (str): Original query

        Returns:
            float: Completeness score between 0 and 1
        """
        query_words = query.lower().split()
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']

        # Identify question types in the query
        question_types = [word for word in question_words if word in query_words]

        if not question_types:
            return 0.7  # Neutral score for non-question queries

        response_lower = response.lower()

        # Expected answer components for different question types
        completeness_indicators = {
            'what': ['is', 'are', 'definition', 'means', 'refers', 'include'],
            'how': ['by', 'through', 'method', 'process', 'way', 'steps'],
            'why': ['because', 'due to', 'reason', 'cause', 'since', 'leads'],
            'when': ['time', 'date', 'period', 'during', 'year', 'ago'],
            'where': ['location', 'place', 'in', 'at', 'within', 'region'],
            'who': ['person', 'people', 'organization', 'company', 'team']
        }

        addressed_aspects = 0
        for q_type in question_types:
            if q_type in completeness_indicators:
                indicators = completeness_indicators[q_type]
                if any(indicator in response_lower for indicator in indicators):
                    addressed_aspects += 1

        return addressed_aspects / len(question_types) if question_types else 0.7

    def calculate_confidence_score(self, response):
        """
        Assess appropriate confidence and uncertainty handling

        Args:
            response (str): Generated response

        Returns:
            float: Confidence balance score between 0 and 1
        """
        response_lower = response.lower()

        # Different types of confidence indicators
        confidence_words = ['clearly', 'definitely', 'certainly', 'obviously', 'precisely']
        uncertainty_words = ['likely', 'possibly', 'might', 'could', 'suggests', 'indicates', 'appears', 'seems']
        overconfidence_words = ['always', 'never', 'all', 'none', 'completely', 'absolutely', 'impossible', 'guarantee']

        # Count occurrences
        confidence_count = sum(1 for word in confidence_words if word in response_lower)
        uncertainty_count = sum(1 for word in uncertainty_words if word in response_lower)
        overconfidence_count = sum(1 for word in overconfidence_words if word in response_lower)

        words_count = len(response.split())

        # Calculate ratios
        confidence_ratio = confidence_count / words_count if words_count else 0
        uncertainty_ratio = uncertainty_count / words_count if words_count else 0
        overconfidence_penalty = overconfidence_count / words_count if words_count else 0

        # Optimal balance: moderate confidence + appropriate uncertainty - overconfidence
        balance_score = (confidence_ratio * 0.3 + uncertainty_ratio * 0.5 - overconfidence_penalty * 0.8)

        # Scale and normalize
        return max(0, min(1, balance_score * 10))

    def evaluate_response(self, query, response, context_docs=None):
        """
        Comprehensive evaluation across all implemented metrics

        Args:
            query (str): Original user query
            response (str): Generated response to evaluate
            context_docs (list, optional): Context documents used for generation

        Returns:
            dict: Dictionary containing all metric scores and overall strength
        """
        print("🔍 Evaluating response across all metrics...")

        metrics = {}

        # Core response quality metrics
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(query, response)
        metrics['information_density'] = self.calculate_information_density(response)
        metrics['coherence_score'] = self.calculate_coherence_score(response)
        metrics['specificity_score'] = self.calculate_specificity_score(response)
        metrics['readability_score'] = self.calculate_readability_score(response)
        metrics['comprehensiveness'] = self.calculate_comprehensiveness(response, query)

        # Advanced quality metrics
        metrics['technical_depth'] = self.calculate_technical_depth(response, query)
        metrics['logical_consistency'] = self.calculate_logical_consistency(response)
        metrics['completeness_score'] = self.calculate_completeness_score(response, query)
        metrics['confidence_score'] = self.calculate_confidence_score(response)

        # Context-dependent metrics (RAG-specific)
        if context_docs and any(doc.strip() for doc in context_docs):
            metrics['context_utilization'] = self.calculate_context_utilization(response, context_docs)
            metrics['factual_grounding'] = self.calculate_factual_grounding(response, context_docs)
            metrics['citation_accuracy'] = self.calculate_citation_accuracy(response, context_docs)
            metrics['hallucination_score'] = self.calculate_hallucination_score(response, context_docs)
            metrics['temporal_relevance'] = self.calculate_temporal_relevance(response, query, context_docs)
            metrics['diversity_score'] = self.calculate_diversity_score(response, context_docs)
        else:
            # Default values when no context is available
            metrics['context_utilization'] = 0.0
            metrics['factual_grounding'] = 0.0
            metrics['citation_accuracy'] = 0.0
            metrics['hallucination_score'] = 0.5  # Neutral without context
            metrics['temporal_relevance'] = 0.5
            metrics['diversity_score'] = 0.0

        # Calculate weighted overall strength score
        weights = {
            'semantic_similarity': 0.10,  # How well response matches query intent
            'information_density': 0.08,  # Richness of information content
            'coherence_score': 0.10,  # Logical flow and structure
            'specificity_score': 0.08,  # Concrete details and facts
            'readability_score': 0.06,  # Text accessibility
            'comprehensiveness': 0.10,  # Complete coverage of query
            'context_utilization': 0.08,  # Effective use of provided context
            'factual_grounding': 0.06,  # Alignment with source material
            'technical_depth': 0.08,  # Domain expertise level
            'logical_consistency': 0.06,  # Internal logical coherence
            'completeness_score': 0.08,  # Query requirement fulfillment
            'confidence_score': 0.04,  # Appropriate uncertainty handling
            'citation_accuracy': 0.03,  # Source attribution quality
            'hallucination_score': 0.03,  # Factual accuracy verification
            'temporal_relevance': 0.01,  # Information currency
            'diversity_score': 0.01  # Multi-source integration
        }

        # Ensure weights sum to 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:
            print(f"⚠️ Warning: Weights sum to {total_weight}, not 1.0")

        # Calculate overall strength score
        strength_score = sum(metrics[metric] * weight for metric, weight in weights.items())
        metrics['overall_strength_score'] = strength_score

        print("✅ Response evaluation complete!")
        return metrics


# =============================================================================
# PERPLEXITY API INTEGRATION
# =============================================================================

def call_perplexity(prompt, max_retries=3):
    """
    Make API call to Perplexity with error handling and retry logic

    Args:
        prompt (str): The prompt to send to the API
        max_retries (int): Maximum number of retry attempts

    Returns:
        str or None: API response content or None if failed
    """
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": prompt}]
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.ok:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                print(f"❌ API Error (Attempt {attempt + 1}/{max_retries}): Status {response.status_code}")
                print(f"Response: {response.text}")

                if attempt < max_retries - 1:
                    print("🔄 Retrying in 2 seconds...")
                    time.sleep(2)

        except requests.exceptions.Timeout:
            print(f"⏰ Request timeout (Attempt {attempt + 1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            print(f"🌐 Network error (Attempt {attempt + 1}/{max_retries}): {e}")
        except Exception as e:
            print(f"❌ Unexpected error (Attempt {attempt + 1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            print("🔄 Retrying...")

    print("❌ All API attempts failed")
    return None


# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================

def main():
    """Main execution function with comprehensive error handling"""

    try:
        # Initialize evaluator
        print("🚀 Initializing RAG evaluator...")
        evaluator = RAGEvaluator(model)
        print("✅ Evaluator ready!")

        # Get user input
        print("\n" + "=" * 60)
        sentence = input("🔎 Enter your query sentence: ").strip()

        if not sentence:
            print("❌ Empty query provided. Exiting.")
            return

        print(f"\n🔍 Processing query: \"{sentence}\"")

        # Generate embeddings for vector search
        try:
            embedding_vector = model.encode(sentence, normalize_embeddings=True).tolist()
        except Exception as e:
            print(f"❌ Failed to generate embeddings: {e}")
            return

        # Perform vector search in MongoDB
        print("📊 Performing vector search...")

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embeddings",
                    "path": "embedding",
                    "queryVector": embedding_vector,
                    "numCandidates": 100,
                    "limit": 5
                }
            }
        ]

        try:
            results = list(collection.aggregate(pipeline))
            print(f"📄 Retrieved {len(results)} documents for query analysis\n")
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
            return

        # Process retrieved documents
        context_snippets = []
        context_texts = []

        for i, doc in enumerate(results, 1):
            title = doc.get("title", f"Document {i}")
            date = doc.get("date", "Unknown date")
            content = doc.get("text") or doc.get("full_text") or doc.get("abstract") or doc.get("content", "")

            if content.strip():  # Only include documents with content
                snippet = f"Title: {title}\nDate: {date}\nContent: {content.strip()}"
                context_snippets.append(snippet)
                context_texts.append(content.strip())

        if not context_texts:
            print("⚠️ No usable documents found. Continuing with comparison...")

        combined_context = "\n\n---\n\n".join(context_snippets)

        # Prepare prompts for comparison
        prompt_with_docs = f"""
You are a senior research analyst working on synthesizing findings across multiple knowledge sources. You are given {len(context_snippets)} documents that were retrieved from a vector similarity search engine for the query:

"{sentence}"

These documents may include reports, academic abstracts, internal notes, or market observations. Your goal is to synthesize the *key ideas* expressed in these documents and produce a unified, coherent, and sophisticated paragraph.

### Constraints:

- Only use information explicitly stated in the documents below.
- Do NOT add any external knowledge, assumptions, or hallucinated facts.
- Avoid bullet points or listing each document separately.
- Do NOT refer to document titles or metadata (e.g., "Document 1 says…").
- Maintain a neutral, analytical tone.
- Prioritize conceptual coherence and interpretive synthesis over summarization.

### Output:

Write a single, well-structured paragraph that:
- Combines themes or patterns across documents.
- Identifies nuanced insights, emerging trends, or common conclusions.
- Reflects critical analysis as if written by a domain expert.

### Documents:
{combined_context}

### Response:
"""

        prompt_without_docs = f"""
You are a synthesis expert. Write a thoughtful, insightful paragraph about the topic: "{sentence}".
Do not use the documents and just treat it as an independent prompt and generate it separately. It is merely used for comparison purposes.
Note: It is very important you treat this as a separate query and do not include the documents mentioned in other prompts since it is used for comparison purposes.
"""

        # Generate responses
        print("🧠 Generating LLM synthesis WITH document context...")
        with_docs_output = call_perplexity(prompt_with_docs)

        print("🧠 Generating LLM synthesis WITHOUT document context...")
        without_docs_output = call_perplexity(prompt_without_docs)

        # Display generated responses
        print("\n" + "=" * 80)
        print("📝 GENERATED RESPONSES")
        print("=" * 80)

        if with_docs_output:
            print("\n🔍 LLM-Synthesized Insight (WITH Documents):\n")
            print(with_docs_output)
        else:
            print("\n❌ Failed to generate response with documents")

        if without_docs_output:
            print("\n🧠 LLM-Synthesized Insight (WITHOUT Documents):\n")
            print(without_docs_output)
        else:
            print("\n❌ Failed to generate response without documents")

        # Perform comprehensive evaluation
        if with_docs_output and without_docs_output:
            print("\n" + "=" * 80)
            print("📊 COMPREHENSIVE RESPONSE EVALUATION")
            print("=" * 80)

            # Evaluate both responses
            with_docs_metrics = evaluator.evaluate_response(sentence, with_docs_output, context_texts)
            without_docs_metrics = evaluator.evaluate_response(sentence, without_docs_output)

            # Display detailed comparison
            display_evaluation_results(with_docs_metrics, without_docs_metrics)

        else:
            print("\n⚠️ Cannot perform evaluation - one or both responses failed to generate")

    except KeyboardInterrupt:
        print("\n\n⏹️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in main execution: {e}")
        import traceback
        traceback.print_exc()


def display_evaluation_results(with_docs_metrics, without_docs_metrics):
    """
    Display comprehensive evaluation results with detailed analysis

    Args:
        with_docs_metrics (dict): Metrics for document-enhanced response
        without_docs_metrics (dict): Metrics for standalone response
    """

    print("\n📈 DETAILED METRICS COMPARISON:")
    print("-" * 70)

    # Metric display names with emojis for better readability
    metric_names = {
        'semantic_similarity': '🎯 Semantic Similarity to Query',
        'information_density': '📊 Information Density',
        'coherence_score': '🔗 Text Coherence',
        'specificity_score': '🔍 Specificity & Detail Level',
        'readability_score': '📖 Readability Score',
        'comprehensiveness': '📋 Comprehensiveness',
        'context_utilization': '📚 Context Utilization',
        'factual_grounding': '⚓ Factual Grounding',
        'technical_depth': '🔬 Technical Depth',
        'logical_consistency': '🧠 Logical Consistency',
        'completeness_score': '✅ Answer Completeness',
        'confidence_score': '🎯 Confidence Balance',
        'citation_accuracy': '📝 Citation Accuracy',
        'hallucination_score': '🚫 Hallucination Prevention',
        'temporal_relevance': '⏰ Temporal Relevance',
        'diversity_score': '🌈 Source Diversity',
        'overall_strength_score': '🏆 OVERALL STRENGTH SCORE'
    }

    improvements = {}

    # Calculate and display individual metric comparisons
    for metric, display_name in metric_names.items():
        with_score = with_docs_metrics[metric]
        without_score = without_docs_metrics[metric]
        improvement = ((with_score - without_score) / max(without_score, 0.001)) * 100
        improvements[metric] = improvement

        print(f"\n{display_name}:")
        print(f"  📚 With Documents:    {with_score:.3f}")
        print(f"  🤖 Without Documents: {without_score:.3f}")
        print(f"  📊 Improvement:       {improvement:+.1f}%")

        # Visual indicator of improvement
        if improvement > 5:
            print(f"  ✅ Significantly better with documents")
        elif improvement > 0:
            print(f"  👍 Better with documents")
        elif improvement < -5:
            print(f"  ❌ Significantly better without documents")
        elif improvement < 0:
            print(f"  👎 Better without documents")
        else:
            print(f"  ➖ No significant difference")

    # Summary statistics
    print("\n" + "=" * 70)
    print("🎯 SUMMARY ANALYSIS:")
    print("=" * 70)

    positive_improvements = [imp for imp in improvements.values() if imp > 0]
    negative_improvements = [imp for imp in improvements.values() if imp < 0]
    significant_positive = [imp for imp in improvements.values() if imp > 5]
    significant_negative = [imp for imp in improvements.values() if imp < -5]

    total_metrics = len(improvements)

    print(f"📊 Total metrics evaluated: {total_metrics}")
    print(f"📈 Metrics where WITH-docs performed better: {len(positive_improvements)}/{total_metrics}")
    print(f"📉 Metrics where WITHOUT-docs performed better: {len(negative_improvements)}/{total_metrics}")
    print(f"🚀 Significant improvements with docs: {len(significant_positive)}")
    print(f"⚠️ Significant degradations with docs: {len(significant_negative)}")
    print(f"📊 Average improvement: {np.mean(list(improvements.values())):.1f}%")
    print(f"📊 Maximum improvement: {max(improvements.values()):.1f}%")
    print(f"📊 Minimum improvement: {min(improvements.values()):.1f}%")

    # Overall verdict
    overall_improvement = improvements['overall_strength_score']
    print(f"\n🏆 OVERALL STRENGTH IMPROVEMENT: {overall_improvement:+.1f}%")

    # Verdict with detailed categorization
    if overall_improvement > 15:
        print("🎉 VERDICT: Document-enhanced response is DRAMATICALLY BETTER!")
        verdict_emoji = "🎉"
    elif overall_improvement > 10:
        print("🌟 VERDICT: Document-enhanced response is SIGNIFICANTLY BETTER!")
        verdict_emoji = "🌟"
    elif overall_improvement > 5:
        print("✅ VERDICT: Document-enhanced response is MODERATELY BETTER!")
        verdict_emoji = "✅"
    elif overall_improvement > 0:
        print("👍 VERDICT: Document-enhanced response is SLIGHTLY BETTER!")
        verdict_emoji = "👍"
    elif overall_improvement > -5:
        print("➖ VERDICT: Minimal difference between approaches!")
        verdict_emoji = "➖"
    else:
        print("⚠️ VERDICT: Document enhancement may have degraded response quality!")
        verdict_emoji = "⚠️"

    # Detailed insights and recommendations
    print("\n💡 KEY INSIGHTS AND RECOMMENDATIONS:")
    print("-" * 50)

    best_metric = max(improvements.items(), key=lambda x: x[1])
    worst_metric = min(improvements.items(), key=lambda x: x[1])

    print(f"🏅 Biggest improvement: {metric_names[best_metric[0]]} (+{best_metric[1]:.1f}%)")
    print(f"🔧 Area needing attention: {metric_names[worst_metric[0]]} ({worst_metric[1]:+.1f}%)")

    # Context utilization analysis
    context_util = with_docs_metrics['context_utilization']
    if context_util > 0.7:
        print("📚 Excellent context utilization - documents were well integrated")
    elif context_util > 0.4:
        print("📖 Good context utilization - some room for better document integration")
    else:
        print("📋 Poor context utilization - consider improving prompt or document relevance")

    # Hallucination analysis
    hallucination = with_docs_metrics['hallucination_score']
    if hallucination > 0.8:
        print("🛡️ Excellent factual accuracy - minimal hallucination risk")
    elif hallucination > 0.6:
        print("✅ Good factual grounding with some verification needed")
    else:
        print("⚠️ Potential hallucination concerns - verify factual claims")

    # Technical depth analysis
    tech_depth = with_docs_metrics['technical_depth']
    if tech_depth > 0.7:
        print("🔬 High technical depth - demonstrates domain expertise")
    elif tech_depth > 0.4:
        print("⚙️ Moderate technical content - appropriate for general audience")
    else:
        print("📝 Basic technical level - may need more specialized terminology")

    print(f"\n{verdict_emoji} Final Assessment: Document-enhanced RAG system shows ")
    if overall_improvement > 5:
        print("strong evidence of improved response quality across multiple dimensions.")
    elif overall_improvement > 0:
        print("modest improvements with room for optimization.")
    else:
        print("mixed results - system configuration may need adjustment.")


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting Enhanced RAG Evaluation System")
    print("=" * 60)

    # Add missing import for time (needed for retry logic)
    import time

    main()

    print("\n" + "=" * 60)
    print("✅ RAG Evaluation Complete!")
