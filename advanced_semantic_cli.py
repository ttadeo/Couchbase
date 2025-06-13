#
# Couchbase Technical Challenge - Advanced Semantic Cache CLI
#
# This professional-grade CLI tool handles both data enrichment and hybrid search.
# It is designed to be flexible, allowing for experimentation with different
# embedding models to fulfill the bonus requirements of the challenge.
#

import argparse
from datetime import timedelta
import sys

# SentenceTransformers is used to generate vector embeddings from text.
from sentence_transformers import SentenceTransformer

# Import necessary components from the Couchbase Python SDK.
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import CouchbaseException
from couchbase.options import ClusterOptions, SearchOptions
from couchbase.search import MatchQuery, SearchRequest
from couchbase.vector_search import VectorQuery, VectorSearch


# --- 1. Configuration (Capella) ---
# This configuration is shared across all commands.
# Update these placeholders with your Capella connection details.
CAPELLA_ENDPOINT = "cb.aa5iafvhxvmxutqg.cloud.couchbase.com"
DB_USERNAME = "capella_app_user"
DB_PASSWORD = "xxxxxxxx" # Your Capella user password
BUCKET_NAME = "travel-sample"
SCOPE_NAME = "inventory"
COLLECTION_NAME = "hotel"


def connect_to_couchbase():
    """Establishes and verifies a connection to the Couchbase Capella cluster."""
    print(f"Connecting to Couchbase Capella at '{CAPELLA_ENDPOINT}'...")
    try:
        auth = PasswordAuthenticator(DB_USERNAME, DB_PASSWORD)
        cluster = Cluster(f"couchbases://{CAPELLA_ENDPOINT}", ClusterOptions(auth))
        cluster.wait_until_ready(timedelta(seconds=10))
        print("Successfully connected to Couchbase Capella.")
        return cluster
    except (CouchbaseException, Exception) as e:
        print(f"Error: Could not connect to Capella. Please check credentials and Allowed IPs. \nDetails: {e}", file=sys.stderr)
        return None


def generate_embeddings(args):
    """
    Handles the 'generate' command. Fetches hotel documents, generates
    embeddings using the specified model, and updates the documents.
    """
    print(f"--- Running Embedding Generation for model: {args.model} ---")
    cluster = connect_to_couchbase()
    if not cluster:
        sys.exit(1)

    try:
        collection = cluster.bucket(BUCKET_NAME).scope(SCOPE_NAME).collection(COLLECTION_NAME)

        print(f"Loading embedding model '{args.model}'...")
        model = SentenceTransformer(args.model)
        
        # Dynamically determine the embedding dimension from the loaded model
        embedding_dimension = model.get_sentence_embedding_dimension()
        print(f"Model loaded. Vector dimension detected: {embedding_dimension}")
        print("\nIMPORTANT: When you create your FTS index, you MUST set the vector dimension to this value.")

        # The field where the new vector embedding will be stored.
        # We include the model name to allow multiple embeddings per document.
        target_embedding_field = f"embedding_{args.model.replace('/', '_')}"
        print(f"Embeddings will be stored in the field: '{target_embedding_field}'\n")

        query = f"SELECT RAW meta(h).id FROM `{BUCKET_NAME}`.`{SCOPE_NAME}`.`{COLLECTION_NAME}` h"
        result = cluster.query(query)
        doc_ids = [row for row in result]
        print(f"Found {len(doc_ids)} hotel documents to process.")

        for i, doc_id in enumerate(doc_ids):
            doc = collection.get(doc_id).content_as[dict]
            text_to_embed = doc.get('description')
            if text_to_embed and isinstance(text_to_embed, str):
                embedding = model.encode(text_to_embed).tolist()
                doc[target_embedding_field] = embedding
                collection.upsert(doc_id, doc)
                print(f"({i+1}/{len(doc_ids)}) Updated document: {doc_id}")
            else:
                print(f"({i+1}/{len(doc_ids)}) Skipped document (missing description): {doc_id}")

        print("\nEmbedding generation and document updates complete.")

    except Exception as e:
        print(f"An unexpected error occurred during embedding generation: {e}", file=sys.stderr)
        sys.exit(1)


def search_hybrid(args):
    """
    Handles the 'search' command. Performs a hybrid search using the
    specified query, model, and FTS index.
    """
    print(f"--- Running Hybrid Search on index: {args.index} ---")
    cluster = connect_to_couchbase()
    if not cluster:
        sys.exit(1)

    try:
        scope = cluster.bucket(BUCKET_NAME).scope(SCOPE_NAME)
        collection = scope.collection(COLLECTION_NAME)

        print(f"Loading query model '{args.model}' to generate vector...")
        model = SentenceTransformer(args.model)
        query_vector = model.encode(args.query).tolist()
        print("Vector generated.")

        # Dynamically set the vector field name based on the model used for the query
        vector_field_name = f"embedding_{args.model.replace('/', '_')}"
        print(f"Querying against vector field: '{vector_field_name}'")

        keyword_query = MatchQuery(args.query, field="description", boost=0.5)
        vector_query = VectorQuery(vector_field_name, query_vector, num_candidates=args.num_results)
        vector_search = VectorSearch.from_vector_query(vector_query)
        search_req = SearchRequest.create(keyword_query).with_vector_search(vector_search)
        
        # We can also add boost options to the search options for more control
        search_options = SearchOptions(limit=args.num_results, fields=["name", "description"])

        print("Executing FTS hybrid search query on Capella 'inventory' scope...")
        result = scope.search(args.index, search_req, search_options)

        print("\n--- Search Results ---")
        rows_found = False
        for row in result.rows():
            rows_found = True
            print(f"Score: {row.score:.4f}")
            doc_fields = row.fields
            if not doc_fields:
                print(f"  (Note: Fetching document '{row.id}' as fields were not stored in index)")
                doc_result = collection.get(row.id)
                doc_fields = doc_result.content_as[dict]
            
            print(f"  Hotel: {doc_fields.get('name', 'N/A')}")
            description = doc_fields.get('description', 'N/A')
            print(f"  Description: {description[:150]}...")
            print("-" * 25)

        if not rows_found:
            print("No results found.")

        metrics = result.metadata().metrics()
        took_ms = metrics.took().total_seconds() * 1000
        print(f"\nQuery completed in {took_ms:.2f} ms")
        print(f"Total Hits Found: {metrics.total_rows()}")

    except Exception as e:
        print(f"An unexpected error occurred during search: {e}", file=sys.stderr)
        sys.exit(1)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Main parser
    parser = argparse.ArgumentParser(description="An advanced semantic search CLI for Couchbase.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Sub-parser for the 'generate' command
    parser_generate = subparsers.add_parser("generate", help="Generate and store embeddings for documents.")
    parser_generate.add_argument(
        "--model",
        default='all-MiniLM-L6-v2',
        help="The Sentence Transformer model to use for generating embeddings (e.g., 'all-MiniLM-L12-v2')."
    )
    parser_generate.set_defaults(func=generate_embeddings)

    # Sub-parser for the 'search' command
    parser_search = subparsers.add_parser("search", help="Perform a hybrid search.")
    parser_search.add_argument(
        "-q", "--query",
        required=True,
        help="The natural language query to search for."
    )
    parser_search.add_argument(
        "-i", "--index",
        required=True,
        help="The name of the FTS index to target."
    )
    parser_search.add_argument(
        "-m", "--model",
        default='all-MiniLM-L6-v2',
        help="The Sentence Transformer model to use for generating the query vector. Must match the model used to build the index."
    )
    parser_search.add_argument(
        "-n", "--num-results",
        type=int,
        default=5,
        help="The number of search results to return."
    )
    parser_search.set_defaults(func=search_hybrid)

    # Parse arguments and call the corresponding function
    args = parser.parse_args()
    args.func(args)
    
    print("\nCLI finished.")

