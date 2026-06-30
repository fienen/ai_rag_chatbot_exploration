# AI RAG Chatbot Exploration

Exploring and testing AI chatbot concepts related to RAG abilities and streaming.

## Development

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key**

   ```bash
   # Windows (PowerShell)
   $env:OPENAI_API_KEY = "sk-..."

   # Windows (Command Prompt)
   set OPENAI_API_KEY=sk-...

   # macOS/Linux
   export OPENAI_API_KEY=sk-...
   ```

3. **Add the source document**

   Place your knowledge base text file at `data/RAG_source.txt`. This file is used to build the vector store for retrieval.

### Running

```bash
python query_transformation_rag.py
```

The script demonstrates four query transformation techniques — Query Rewriting, Query Expansion, Query Decomposition, and HyDE — printing section-separated output to the console for each technique.

### What to expect

Each run will:

1. Load and chunk `data/RAG_source.txt` into a Chroma vector store using OpenAI embeddings.
2. Run a baseline retrieval with a short query to show the problem.
3. Show how a manually improved query changes the retrieved chunks.
4. Call the OpenAI API (model `gpt-4o`) four times to demonstrate each transformation technique and print the results.

API calls are made on every run — expect a few seconds of latency and associated token usage.

