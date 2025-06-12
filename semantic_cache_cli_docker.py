#
# Couchbase Technical Challenge - Semantic Cache CLI (Docker & FTS Version)
#
# This application connects to a local Couchbase Docker instance and performs a
# hybrid search using the Full-Text Search (FTS) service. It combines keyword
# and vector search to find the most relevant results from the 'travel-sample' dataset.
#

import argparse
from datetime import timedelta

# SentenceTransformers is used to generate vector embeddings from text.
from sentence_transformers import SentenceTransformer

# Import necessary components from the Couchbase Python SDK.
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import CouchbaseException
# Import the specific search query builder classes.
from couchbase.options import ClusterOptions, SearchOptions
from couchbase.search import MatchQuery


# --- 1. Configuration (Local Docker) ---
# Update these placeholders with your local Couchbase Docker instance details.
DB_HOST = "localhost"
# The administrator credentials for your local Docker instance.
DB_USERNAME = "app_user"
DB_PASSWORD = "Sylveo$259"
# The name of the FTS index we created in the local Couchbase UI.
INDEX_NAME = "hybrid-hotel-search-index"

# The embedding model to use.
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


def connect_to_couchbase():
    """
    Establishes and verifies a connection to the local Couchbase cluster.
    Returns a Cluster object on success, or None on failure.
    """
    print(f"Connecting to local Couchbase instance at '{DB_HOST}'...")
    try:
        # Authenticate using the database credentials.
        auth = PasswordAuthenticator(DB_USERNAME, DB_PASSWORD)
        # Connect to the cluster using a non-TLS connection string for local instances.
        cluster = Cluster(f"couchbase://{DB_HOST}", ClusterOptions(auth))
        # Wait up to 5 seconds to verify the connection is ready.
        cluster.wait_until_ready(timedelta(seconds=5))
        print("Successfully connected to local Couchbase instance.")
        return cluster
    except (CouchbaseException, Exception) as e:
        print(f"Error: Could not connect to local Couchbase. Please ensure Docker is running. \nDetails: {e}")
        return None


def hybrid_search(cluster, query_text, num_results=5):
    """
    Performs a hybrid search using the FTS service.

    Args:
        cluster: The connected Couchbase Cluster object.
        query_text: The user's search query string.
        num_results: The number of results to return.
    """
    if not cluster:
        return

    print(f"\nPerforming hybrid search for: '{query_text}'...")

    try:
        # Generate the vector for the user's query.
        print("Generating query vector...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        query_vector = model.encode(query_text).tolist()
        print("Vector generated.")

        # --- Define the Hybrid Search Query using FTS ---

        # 1. Define the keyword search portion of the query.
        keyword_query = MatchQuery(query_text, field="description", boost=0.5)

        # 2. Define the vector search portion as a dictionary for the knn parameter.
        vector_search_opts = {
            "field": "description_embedding",
            "vector": query_vector,
            "k": num_results,
            "boost": 1.5
        }

        # 3. Define SearchOptions to control the output and include the vector search.
        search_options = SearchOptions(
            limit=num_results,
            fields=["name", "description"],
            knn=[vector_search_opts] # Pass the vector search config here
        )

        # 4. Execute the search query with the keyword query and combined options.
        print("Executing FTS hybrid search query...")
        result = cluster.search_query(
            INDEX_NAME,
            keyword_query,
            search_options
        )

        # --- Display the Results ---
        print("\n--- Search Results ---")
        rows_found = False
        for row in result.rows():
            rows_found = True
            print(f"Score: {row.score:.4f}")
            print(f"  Hotel: {row.fields.get('name', 'N/A')}")
            # For cleaner output, truncate long descriptions.
            description = row.fields.get('description', 'N/A')
            print(f"  Description: {description[:150]}...")
            print("-" * 25)

        if not rows_found:
            print("No results found.")

        # Display performance metrics from the search.
        metrics = result.meta_data().metrics
        print(f"\nQuery completed in {metrics.took() / 1_000_000:.2f} ms")
        print(f"Total Hits Found: {metrics.total_rows()}")

    except CouchbaseException as e:
        print(f"An FTS query error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A command-line semantic search tool for a local Couchbase Docker instance.",
        epilog="Example: python semantic_cache_cli_docker.py --query \"a hotel with a nice garden\""
    )
    parser.add_argument(
        "-q",
        "--query",
        required=True,
        help="The search query text to execute."
    )
    args = parser.parse_args()

    couchbase_cluster = connect_to_couchbase()

    if couchbase_cluster:
        hybrid_search(couchbase_cluster, args.query)
        print("\nCLI finished.")
