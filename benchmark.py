"""
Benchmark — Ground-truth Q&A dataset for evaluating the RAG pipeline.

Each entry has:
- question: The test question
- expected_answer_keywords: Key phrases that a correct answer MUST contain
- expected_source: The document that should be retrieved
- category: Type of test (factual, cross-doc, abstention, specificity)
- difficulty: easy / medium / hard
"""

BENCHMARK = [
    # ── Factual questions (clear answer in one document) ──
    {
        "id": "F1",
        "question": "What are the three types of machine learning?",
        "expected_answer_keywords": ["supervised", "unsupervised", "reinforcement"],
        "expected_source": "artificial_intelligence.txt",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "F2",
        "question": "How much has global temperature increased since pre-industrial levels?",
        "expected_answer_keywords": ["1.1"],
        "expected_source": "climate_change.txt",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "F3",
        "question": "Who was the first human in space?",
        "expected_answer_keywords": ["yuri gagarin", "gagarin"],
        "expected_source": "space_exploration.txt",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "F4",
        "question": "What is Shor's algorithm used for?",
        "expected_answer_keywords": ["factorization", "integer"],
        "expected_source": "quantum_computing.pdf",
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "id": "F5",
        "question": "What consensus mechanisms does blockchain use?",
        "expected_answer_keywords": ["proof of work", "proof of stake"],
        "expected_source": "blockchain.docx",
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "id": "F6",
        "question": "What year was the Mars rover Perseverance landed?",
        "expected_answer_keywords": ["2021"],
        "expected_source": "space_exploration.txt",
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "id": "F7",
        "question": "What is the surface code in quantum computing?",
        "expected_answer_keywords": ["error correction", "qec", "qubit"],
        "expected_source": "quantum_computing.pdf",
        "category": "factual",
        "difficulty": "hard",
    },
    {
        "id": "F8",
        "question": "What are common algorithms used in supervised learning?",
        "expected_answer_keywords": ["decision tree", "random forest", "svm", "neural network"],
        "expected_source": "artificial_intelligence.txt",
        "category": "factual",
        "difficulty": "medium",
    },

    # ── Specificity (requires precise retrieval) ──
    {
        "id": "S1",
        "question": "Who created Bitcoin and when?",
        "expected_answer_keywords": ["satoshi nakamoto", "2009"],
        "expected_source": "blockchain.docx",
        "category": "specificity",
        "difficulty": "easy",
    },
    {
        "id": "S2",
        "question": "What is the greenhouse effect?",
        "expected_answer_keywords": ["trap", "heat", "atmosphere"],
        "expected_source": "climate_change.txt",
        "category": "specificity",
        "difficulty": "medium",
    },

    # ── Abstention (no relevant info in documents) ──
    {
        "id": "A1",
        "question": "What is the recipe for chocolate chip cookies?",
        "expected_answer_keywords": ["cannot answer", "not", "no information", "don't have"],
        "expected_source": None,
        "category": "abstention",
        "difficulty": "easy",
    },
    {
        "id": "A2",
        "question": "What is the capital of France?",
        "expected_answer_keywords": ["cannot answer", "not", "no information", "don't have"],
        "expected_source": None,
        "category": "abstention",
        "difficulty": "easy",
    },
    {
        "id": "A3",
        "question": "How do I train a large language model from scratch?",
        "expected_answer_keywords": ["cannot answer", "not", "no information", "don't have", "insufficient"],
        "expected_source": None,
        "category": "abstention",
        "difficulty": "hard",
    },
]
