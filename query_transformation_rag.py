"""
Query Transformation for RAG Systems
=====================================
Extracted from QueryTransformationRAGSystems.ipynb and converted into a
plain runnable Python script.

This script demonstrates four query transformation techniques for
improving retrieval in a RAG (Retrieval-Augmented Generation) system:

    1. Query Rewriting
    2. Query Expansion
    3. Query Decomposition
    4. HyDE (Hypothetical Document Embeddings)

Requirements:
    - An OpenAI API key set in the OPENAI_API_KEY environment variable.
    - A text file at data/RAG_source.txt containing the pharmaceutical
      knowledge base used for retrieval (see README.md for details).

Run with:
    python query_transformation_rag.py
"""

import os

from openai import OpenAI

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def section(title: str) -> None:
    """Print a simple section header to the console."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    # === Import required libraries / init client ===
    client = OpenAI()

    # === Load pharmaceutical documents ===
    data_path = os.path.join("data", "RAG_source.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find '{data_path}'. Please add the pharmaceutical "
            "source text file to the data/ folder before running this script. "
            "See README.md for details."
        )

    loader = TextLoader(data_path)
    documents = loader.load()

    # === Split into chunks ===
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    chunks = splitter.split_documents(documents)

    # === Create embeddings and vector store ===
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    # === Create retriever ===
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    section("Vector Store Setup Complete")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embedding model: {embeddings.model}")
    print(f"Retriever configured: Top-{retriever.search_kwargs['k']} similarity search")

    # ------------------------------------------------------------------
    # Baseline: The Problem with Short Queries
    # ------------------------------------------------------------------
    short_query = "Is Zelomax safe during pregnancy"
    baseline_results = retriever.invoke(short_query)

    section("Baseline Retrieval Results")
    print(f'Query: "{short_query}"')
    for i, doc in enumerate(baseline_results, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content)

    # ------------------------------------------------------------------
    # Demonstration: improved query phrasing
    # ------------------------------------------------------------------
    improved_query = "What is Zelomax's risk classification for pregnancy?"
    improved_results = retriever.invoke(improved_query)

    section("Query Transformation Results (manually rephrased query)")
    print(f'Query: "{improved_query}"')
    for i, doc in enumerate(improved_results, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content)

    # ------------------------------------------------------------------
    # Technique 1: Query Rewriting
    # ------------------------------------------------------------------
    query_rewrite = "Is Zelomax safe during pregnancy"

    rewrite_prompt = f"""
You are an expert at rewriting users' queries for the purposes of finding information, and linking possible alternative phrasings.

Given the user's original query, generate 3 alternative phrasings that express the same intent in different ways.
The last alternative should try to use some synonyms for keywords, if appropriate.
Each alternative should be a complete, well-formed question or statement.

Original query: {query_rewrite}

Generate only 3 alternative wordings, and then number them 1, 2, and 3:
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": rewrite_prompt}],
    )
    rewrites = response.choices[0].message.content.strip()

    section("Technique 1: Query Rewriting")
    print(f'Original query: "{query_rewrite}"')
    print("\nRewritten alternatives:")
    print(rewrites)

    # ------------------------------------------------------------------
    # Technique 2: Query Expansion
    # ------------------------------------------------------------------
    query_expand = "dosage information"

    expansion_prompt = f"""You are a clinician who is an expert at expanding medical and pharmaceutical queries that users have by
identifying relevant synonyms and related terms that will ensure the user finds the best information even if they don't know
the exactly correct words.

Given the original query, provide a list of:
- Synonyms
- Related medical/pharmaceutical terms
- The most common alternative phrasings for medical contexts
- The most common colloquial alternative phrasings

Original query: {query_expand}

Provide this list of expanded terms as a comma-separated list:
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": expansion_prompt}],
    )
    expanded_terms = response.choices[0].message.content.strip()

    section("Technique 2: Query Expansion")
    print(f'Original query: "{query_expand}"')
    print("\nExpanded terms:")
    print(expanded_terms)

    # ------------------------------------------------------------------
    # Technique 3: Query Decomposition
    # ------------------------------------------------------------------
    query_decompose = """I'm a physician considering Xenthera for my patients with chronic respiratory conditions.
I've read some promising information about its effectiveness, but I need comprehensive details.
What are the primary therapeutic benefits and clinical advantages of prescribing Xenthera for respiratory treatment?
Additionally, what are the specific contraindications, patient populations, or medical conditions where Xenthera should not be prescribed?
Finally, are there any drug interactions or comorbidities I should be aware of before prescribing this medication?
"""

    decomposition_prompt = f"""You are an expert at simplifying users' complex questions and breaking them into smaller sub-questions with a singular focus.

Decompose the original query into 2-5 simpler sub-questions that:
- Attempts to break the original query into a sub-question focused on one subject.
- If there are too many subjects, try to group them logically where they are related.
- Ensure that the entire, complete scope of the original question is addressed
- Are specific and answerable on their own

Original query: {query_decompose}

Provide the sub-questions as a numbered list:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": decomposition_prompt}],
    )
    sub_questions = response.choices[0].message.content.strip()

    section("Technique 3: Query Decomposition")
    print(f'Original complex query:\n"{query_decompose}"')
    print("\nDecomposed sub-questions:")
    print(sub_questions)

    # ------------------------------------------------------------------
    # Technique 4: HyDE (Hypothetical Document Embeddings)
    # ------------------------------------------------------------------
    query_hyde = "What are the safety concerns with Zelomax during pregnancy?"

    hyde_prompt = f"""You are an expert at creating medical and pharmaceutical documentation. Given a user question,
generate a detailed hypothetical document excerpt that would answer it.

Write in a clear and professional manner to create content that would be natural in a medical document with:
- Professional, scientific medical terminology
- Specific, concise details and facts
- Formatting appropriate for a medical journal or study

Question: {query_hyde}

Hypothetical document excerpt:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": hyde_prompt}],
    )
    hypothetical_doc = response.choices[0].message.content.strip()

    # Test retrieval with both approaches
    hyde_retrieval = retriever.invoke(hypothetical_doc)
    baseline_retrieval = retriever.invoke(query_hyde)

    section("Technique 4: HyDE")
    print(f'Original question: "{query_hyde}"')
    print("\nHypothetical document generated:")
    print(hypothetical_doc)

    print("\n--- Retrieval Comparison ---")
    print("\nBaseline (question used directly):")
    print(baseline_retrieval[0].page_content)

    print("\nHyDE (hypothetical document used for retrieval):")
    print(hyde_retrieval[0].page_content)


if __name__ == "__main__":
    main()
