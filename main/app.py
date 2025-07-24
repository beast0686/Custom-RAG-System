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

# Download required NLTK data (UPDATED FIX)
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("📦 Downloading required NLTK data...")
    nltk.download('punkt_tab')  # New punkt tokenizer
    nltk.download('stopwords')
    nltk.download('punkt')  # Fallback for compatibility
    print("✅ NLTK data download complete!")

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
collection = client[MONGO_DB][MONGO_COLLECTION]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


class RAGEvaluator:
    def __init__(self, embedding_model):
        self.model = embedding_model
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("⚠️ Stopwords not found, downloading...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

    def calculate_semantic_similarity(self, query, response):
        """Calculate semantic similarity between query and response"""
        query_embedding = self.model.encode([query])
        response_embedding = self.model.encode([response])
        similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
        return float(similarity)

    def calculate_context_utilization(self, response, context_docs):
        """Calculate how well the response utilizes the provided context"""
        if not context_docs:
            return 0.0

        response_embedding = self.model.encode([response])
        context_embeddings = self.model.encode(context_docs)

        # Calculate average similarity to all context documents
        similarities = cosine_similarity(response_embedding, context_embeddings)[0]
        return float(np.mean(similarities))

    def calculate_information_density(self, text):
        """Calculate information density based on unique concepts and technical terms"""
        try:
            words = word_tokenize(text.lower())
        except LookupError:
            # Fallback to simple split if tokenizer fails
            words = text.lower().split()

        words = [w for w in words if w.isalpha() and w not in self.stop_words]

        if not words:
            return 0.0

        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)

        # Technical term indicators (words longer than 6 characters)
        technical_terms = [w for w in words if len(w) > 6]
        technical_ratio = len(technical_terms) / len(words) if words else 0

        return (unique_ratio + technical_ratio) / 2

    def calculate_coherence_score(self, text):
        """Calculate text coherence based on sentence structure and flow"""
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback to simple sentence splitting
            sentences = text.split('. ')

        if len(sentences) < 2:
            return 0.5  # Neutral score for single sentence

        # Calculate inter-sentence similarity
        sentence_embeddings = self.model.encode(sentences)
        similarities = []

        for i in range(len(sentences) - 1):
            sim = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]
            similarities.append(sim)

        # Good coherence means moderate inter-sentence similarity (not too high, not too low)
        avg_similarity = np.mean(similarities)
        # Optimal coherence is around 0.3-0.7 similarity
        coherence_score = 1 - abs(avg_similarity - 0.5) / 0.5
        return float(coherence_score)

    def calculate_specificity_score(self, text):
        """Calculate how specific and detailed the response is"""
        # Count specific indicators
        specific_indicators = [
            r'\b\d+\b',  # Numbers
            r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',  # Proper nouns (names, places)
            r'\b\d{4}\b',  # Years
            r'\$\d+',  # Dollar amounts
            r'\d+%',  # Percentages
        ]

        total_indicators = 0
        for pattern in specific_indicators:
            total_indicators += len(re.findall(pattern, text))

        try:
            words = len(word_tokenize(text))
        except LookupError:
            words = len(text.split())

        specificity = total_indicators / words if words > 0 else 0
        return min(specificity * 10, 1.0)  # Normalize to 0-1 scale

    def calculate_readability_score(self, text):
        """Calculate readability using multiple metrics"""
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            # Convert Flesch score (0-100) to 0-1 scale, where higher is better
            readability = flesch_score / 100
            return max(0, min(1, readability))
        except:
            return 0.5  # Default neutral score

    def calculate_comprehensiveness(self, response, query):
        """Calculate how comprehensive the response is relative to the query"""
        try:
            query_words = set(word_tokenize(query.lower()))
            response_words = set(word_tokenize(response.lower()))
        except LookupError:
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())

        query_words = {w for w in query_words if w.isalpha() and w not in self.stop_words}
        response_words = {w for w in response_words if w.isalpha() and w not in self.stop_words}

        if not query_words:
            return 0.0

        # Calculate coverage of query concepts in response
        coverage = len(query_words.intersection(response_words)) / len(query_words)

        # Factor in response length (longer responses can be more comprehensive)
        length_factor = min(len(response.split()) / 100, 1.0)  # Normalize around 100 words

        return (coverage + length_factor) / 2

    def calculate_factual_grounding(self, response, context_docs):
        """Estimate how well the response is grounded in provided facts"""
        if not context_docs:
            return 0.0

        try:
            response_words = set(word_tokenize(response.lower()))
        except LookupError:
            response_words = set(response.lower().split())

        response_words = {w for w in response_words if w.isalpha() and w not in self.stop_words}

        context_words = set()
        for doc in context_docs:
            try:
                doc_words = set(word_tokenize(doc.lower()))
            except LookupError:
                doc_words = set(doc.lower().split())
            doc_words = {w for w in doc_words if w.isalpha() and w not in self.stop_words}
            context_words.update(doc_words)

        if not response_words:
            return 0.0

        # Calculate overlap between response and context vocabulary
        overlap = len(response_words.intersection(context_words))
        grounding_score = overlap / len(response_words)

        return grounding_score

    def calculate_citation_accuracy(self, response, context_docs):
        """Measure how well the response attributes information to sources"""
        citation_patterns = [r'\[(\d+)\]', r'\((\d+)\)', r'according to', r'as stated in']
        citations_found = 0
        for pattern in citation_patterns:
            citations_found += len(re.findall(pattern, response))

        # Normalize by response length
        words = len(response.split())
        citation_density = citations_found / max(words, 1) * 100
        return min(citation_density, 1.0)

    def calculate_hallucination_score(self, response, context_docs):
        """Detect potential hallucinations by finding claims not in source docs"""
        if not context_docs:
            return 1.0  # High hallucination risk without context

        # Extract factual claims (sentences with numbers, dates, specific terms)
        factual_patterns = [r'\b\d+%\b', r'\b\d{4}\b', r'\$\d+', r'\b[A-Z][a-z]+ \d+\b']
        response_facts = []
        for pattern in factual_patterns:
            response_facts.extend(re.findall(pattern, response))

        if not response_facts:
            return 0.5  # Neutral if no factual claims

        # Check if facts appear in context
        context_text = ' '.join(context_docs).lower()
        verified_facts = sum(1 for fact in response_facts if fact.lower() in context_text)

        return verified_facts / len(response_facts) if response_facts else 0.5

    def calculate_temporal_relevance(self, response, query, context_docs):
        """Measure how current/up-to-date the information is"""
        import datetime
        current_year = datetime.datetime.now().year

        # Extract years from response and context
        years_in_response = [int(year) for year in re.findall(r'\b(20\d{2})\b', response)]
        years_in_context = []
        for doc in context_docs:
            years_in_context.extend([int(year) for year in re.findall(r'\b(20\d{2})\b', doc)])

        if not years_in_response:
            return 0.5  # Neutral if no temporal info

        # Calculate recency score (prefer more recent years)
        avg_year = np.mean(years_in_response)
        recency_score = max(0, 1 - (current_year - avg_year) / 10)  # Decay over 10 years

        return min(recency_score, 1.0)

    def calculate_diversity_score(self, response, context_docs):
        """Measure how well the response covers diverse aspects from multiple sources"""
        if not context_docs or len(context_docs) < 2:
            return 0.0

        response_embedding = self.model.encode([response])
        doc_embeddings = self.model.encode(context_docs)

        # Calculate similarity to each document
        similarities = cosine_similarity(response_embedding, doc_embeddings)[0]

        # Good diversity means moderate similarity to multiple sources
        # (not too focused on one source, not too scattered)
        diversity_score = 1 - np.std(similarities)  # Lower std = more balanced coverage
        return max(0, min(1, diversity_score))

    def calculate_technical_depth(self, response, query):
        """Measure technical sophistication and domain expertise level"""
        # Technical indicators
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+ly\b',  # Adverbs (often technical descriptions)
            r'\b\w{10,}\b',  # Long technical terms
            r'\b\d+\.\d+\b',  # Version numbers, precise measurements
        ]

        technical_terms = 0
        for pattern in technical_patterns:
            technical_terms += len(re.findall(pattern, response))

        # Domain-specific vocabulary complexity
        words = response.split()
        avg_word_length = np.mean([len(word) for word in words if word.isalpha()])

        # Combine indicators
        technical_density = technical_terms / len(words) if words else 0
        length_complexity = min(avg_word_length / 8, 1.0)  # Normalize around 8 chars

        return (technical_density * 5 + length_complexity) / 2  # Weight technical terms more

    def calculate_logical_consistency(self, response):
        """Detect logical contradictions and inconsistencies in the response"""
        sentences = sent_tokenize(response) if hasattr(self, 'sent_tokenize') else response.split('.')

        if len(sentences) < 2:
            return 1.0  # Perfect consistency for single sentence

        # Look for contradiction indicators
        contradiction_words = ['however', 'but', 'although', 'despite', 'whereas', 'contradicts']
        negation_words = ['not', 'no', 'never', 'neither', 'none']

        contradiction_score = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            contradictions = sum(1 for word in contradiction_words if word in sentence_lower)
            negations = sum(1 for word in negation_words if word in sentence_lower)
            contradiction_score += (contradictions + negations * 0.5)

        # Normalize - some contradictions are natural in complex topics
        normalized_score = 1 - min(contradiction_score / len(sentences), 0.5)
        return max(0, normalized_score)

    def calculate_completeness_score(self, response, query):
        """Measure how completely the response addresses all aspects of the query"""
        # Extract key question words and topics
        query_words = query.lower().split()
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']

        # Identify question type and expected answer components
        question_types = [word for word in question_words if word in query_words]

        if not question_types:
            return 0.7  # Neutral for non-question queries

        # Check if response addresses each question type appropriately
        response_lower = response.lower()
        completeness_indicators = {
            'what': ['is', 'are', 'definition', 'means', 'refers'],
            'how': ['by', 'through', 'method', 'process', 'way'],
            'why': ['because', 'due to', 'reason', 'cause', 'since'],
            'when': ['time', 'date', 'period', 'during', 'year'],
            'where': ['location', 'place', 'in', 'at', 'within'],
            'who': ['person', 'people', 'organization', 'company']
        }

        addressed_aspects = 0
        for q_type in question_types:
            if q_type in completeness_indicators:
                indicators = completeness_indicators[q_type]
                if any(indicator in response_lower for indicator in indicators):
                    addressed_aspects += 1

        return addressed_aspects / len(question_types) if question_types else 0.7

    def calculate_confidence_score(self, response):
        """Measure how appropriately the response handles uncertainty"""
        # Confidence indicators (positive)
        confidence_words = ['clearly', 'definitely', 'certainly', 'obviously', 'precisely']

        # Uncertainty indicators (shows appropriate caution)
        uncertainty_words = ['likely', 'possibly', 'might', 'could', 'suggests', 'indicates', 'appears']

        # Overconfidence indicators (negative)
        overconfidence_words = ['always', 'never', 'all', 'none', 'completely', 'absolutely']

        response_lower = response.lower()

        confidence_count = sum(1 for word in confidence_words if word in response_lower)
        uncertainty_count = sum(1 for word in uncertainty_words if word in response_lower)
        overconfidence_count = sum(1 for word in overconfidence_words if word in response_lower)

        words = len(response.split())

        # Good confidence balance: some certainty, appropriate uncertainty, minimal overconfidence
        confidence_ratio = confidence_count / words
        uncertainty_ratio = uncertainty_count / words
        overconfidence_penalty = overconfidence_count / words

        # Optimal balance: moderate confidence + appropriate uncertainty - overconfidence
        balance_score = (confidence_ratio * 0.3 + uncertainty_ratio * 0.5 - overconfidence_penalty * 0.8)
        return max(0, min(1, balance_score * 10))  # Scale up and normalize


    def evaluate_response(self, query, response, context_docs=None):
        """Comprehensive evaluation of a response with all metrics"""
        metrics = {}

        # Original metrics
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(query, response)
        metrics['information_density'] = self.calculate_information_density(response)
        metrics['coherence_score'] = self.calculate_coherence_score(response)
        metrics['specificity_score'] = self.calculate_specificity_score(response)
        metrics['readability_score'] = self.calculate_readability_score(response)
        metrics['comprehensiveness'] = self.calculate_comprehensiveness(response, query)

        # NEW ADVANCED METRICS
        metrics['technical_depth'] = self.calculate_technical_depth(response, query)
        metrics['logical_consistency'] = self.calculate_logical_consistency(response)
        metrics['completeness_score'] = self.calculate_completeness_score(response, query)
        metrics['confidence_score'] = self.calculate_confidence_score(response)

        # Context-dependent metrics
        if context_docs:
            metrics['context_utilization'] = self.calculate_context_utilization(response, context_docs)
            metrics['factual_grounding'] = self.calculate_factual_grounding(response, context_docs)
            metrics['citation_accuracy'] = self.calculate_citation_accuracy(response, context_docs)
            metrics['hallucination_score'] = self.calculate_hallucination_score(response, context_docs)
            metrics['temporal_relevance'] = self.calculate_temporal_relevance(response, query, context_docs)
            metrics['diversity_score'] = self.calculate_diversity_score(response, context_docs)
        else:
            metrics['context_utilization'] = 0.0
            metrics['factual_grounding'] = 0.0
            metrics['citation_accuracy'] = 0.0
            metrics['hallucination_score'] = 0.5  # Neutral without context
            metrics['temporal_relevance'] = 0.5
            metrics['diversity_score'] = 0.0

        # Updated weights for overall score
        weights = {
            'semantic_similarity': 0.10,
            'information_density': 0.08,
            'coherence_score': 0.10,
            'specificity_score': 0.08,
            'readability_score': 0.06,
            'comprehensiveness': 0.10,
            'context_utilization': 0.08,
            'factual_grounding': 0.06,
            'technical_depth': 0.08,
            'logical_consistency': 0.06,
            'completeness_score': 0.08,
            'confidence_score': 0.04,
            'citation_accuracy': 0.03,
            'hallucination_score': 0.03,
            'temporal_relevance': 0.01,
            'diversity_score': 0.01
        }

        strength_score = sum(metrics[metric] * weight for metric, weight in weights.items())
        metrics['overall_strength_score'] = strength_score

        return metrics


# Initialize evaluator
evaluator = RAGEvaluator(model)

# User input
sentence = input("🔎 Enter your query sentence: ").strip()
embedding_vector = model.encode(sentence, normalize_embeddings=True).tolist()

# Vector search in MongoDB
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
results = list(collection.aggregate(pipeline))
print(f"\n📄 Retrieved {len(results)} documents for: \"{sentence}\"\n")

# Build context from results
context_snippets = []
context_texts = []  # For evaluation
for doc in results:
    title = doc.get("title", "Untitled")
    date = doc.get("date", "Unknown")
    content = doc.get("text") or doc.get("full_text") or doc.get("abstract") or ""
    snippet = f"Title: {title}\nDate: {date}\nContent: {content.strip()}"
    context_snippets.append(snippet)
    context_texts.append(content.strip())

combined_context = "\n\n---\n\n".join(context_snippets)

# Prompts
prompt_with_docs = f"""
You are a senior research analyst working on synthesizing findings across multiple knowledge sources. You are given 5 documents that were retrieved from a vector similarity search engine for the query:

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
Do not use the documents and just treat it as an independant prompt and generate it seperately. It is merely used for comparasion purposes.
Note: It is very important you treat this as a seperate query and do not include the documents mentioned in other prompts since it is used for comparision purposes.
"""


# Helper function to call Perplexity
def call_perplexity(prompt):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print(f"❌ API Error ({response.status_code}): {response.text}")
        return None


# Get responses
print("\n🧠 Generating LLM synthesis WITH document context...")
with_docs_output = call_perplexity(prompt_with_docs)

print("\n🧠 Generating LLM synthesis WITHOUT document context...")
without_docs_output = call_perplexity(prompt_without_docs)

# Display results
if with_docs_output:
    print("\n🔍 LLM-Synthesized Insight (WITH Documents):\n")
    print(with_docs_output)

if without_docs_output:
    print("\n🧠 LLM-Synthesized Insight (WITHOUT Documents):\n")
    print(without_docs_output)

# Evaluate both responses
if with_docs_output and without_docs_output:
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE RESPONSE EVALUATION")
    print("=" * 80)

    # Evaluate response with documents
    with_docs_metrics = evaluator.evaluate_response(sentence, with_docs_output, context_texts)

    # Evaluate response without documents
    without_docs_metrics = evaluator.evaluate_response(sentence, without_docs_output)

    # Display detailed metrics comparison
    print("\n📈 DETAILED METRICS COMPARISON:")
    print("-" * 60)

    metric_names = {
        'semantic_similarity': 'Semantic Similarity to Query',
        'information_density': 'Information Density',
        'coherence_score': 'Text Coherence',
        'specificity_score': 'Specificity & Detail Level',
        'readability_score': 'Readability Score',
        'comprehensiveness': 'Comprehensiveness',
        'context_utilization': 'Context Utilization',
        'factual_grounding': 'Factual Grounding',
        'technical_depth': '🔬 Technical Depth',
        'logical_consistency': '🧠 Logical Consistency',
        'completeness_score': '✅ Answer Completeness',
        'confidence_score': '🎯 Confidence Balance',
        'citation_accuracy': '📚 Citation Accuracy',
        'hallucination_score': '🚫 Hallucination Prevention',
        'temporal_relevance': '⏰ Temporal Relevance',
        'diversity_score': '🌈 Source Diversity',
        'overall_strength_score': '🏆 OVERALL STRENGTH SCORE'
    }

    improvements = {}

    for metric, display_name in metric_names.items():
        with_score = with_docs_metrics[metric]
        without_score = without_docs_metrics[metric]
        improvement = ((with_score - without_score) / max(without_score, 0.001)) * 100
        improvements[metric] = improvement

        print(f"\n{display_name}:")
        print(f"  📚 With Documents:    {with_score:.3f}")
        print(f"  🤖 Without Documents: {without_score:.3f}")
        print(f"  📊 Improvement:       {improvement:+.1f}%")

        if improvement > 0:
            print(f"  ✅ Better with documents")
        elif improvement < 0:
            print(f"  ❌ Better without documents")
        else:
            print(f"  ➖ No difference")

    # Summary statistics
    print("\n" + "=" * 60)
    print("🎯 SUMMARY ANALYSIS:")
    print("=" * 60)

    positive_improvements = [imp for imp in improvements.values() if imp > 0]
    negative_improvements = [imp for imp in improvements.values() if imp < 0]

    print(f"📊 Metrics where WITH-docs performed better: {len(positive_improvements)}/8")
    print(f"📊 Metrics where WITHOUT-docs performed better: {len(negative_improvements)}/8")
    print(f"📊 Average improvement with documents: {np.mean(list(improvements.values())):.1f}%")
    print(f"📊 Maximum improvement: {max(improvements.values()):.1f}%")
    print(f"📊 Minimum improvement: {min(improvements.values()):.1f}%")

    # Overall verdict
    overall_improvement = improvements['overall_strength_score']
    print(f"\n🏆 OVERALL STRENGTH IMPROVEMENT: {overall_improvement:+.1f}%")

    if overall_improvement > 10:
        print("🎉 VERDICT: Document-enhanced response is SIGNIFICANTLY BETTER!")
    elif overall_improvement > 5:
        print("✅ VERDICT: Document-enhanced response is MODERATELY BETTER!")
    elif overall_improvement > 0:
        print("👍 VERDICT: Document-enhanced response is SLIGHTLY BETTER!")
    else:
        print("⚠️  VERDICT: Document enhancement did not improve response quality!")

    # Recommendations
    print("\n💡 KEY INSIGHTS:")
    best_metric = max(improvements.items(), key=lambda x: x[1])
    worst_metric = min(improvements.items(), key=lambda x: x[1])

    print(f"   • Biggest improvement: {metric_names[best_metric[0]]} (+{best_metric[1]:.1f}%)")
    print(f"   • Area for improvement: {metric_names[worst_metric[0]]} ({worst_metric[1]:+.1f}%)")

    if with_docs_metrics['context_utilization'] > 0.7:
        print("   • Excellent context utilization - documents were well integrated")
    elif with_docs_metrics['context_utilization'] > 0.4:
        print("   • Good context utilization - some room for better document integration")
    else:
        print("   • Poor context utilization - consider improving prompt or document relevance")
