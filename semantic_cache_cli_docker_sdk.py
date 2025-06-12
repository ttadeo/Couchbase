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
# Import the specific search query builder classes, including the SearchRequest.
from couchbase.options import ClusterOptions, SearchOptions
from couchbase.search import MatchQuery, SearchRequest
from couchbase.vector_search import VectorQuery, VectorSearch


# --- 1. Configuration (Local Docker) ---
# Update these placeholders with your local Couchbase Docker instance details.
DB_HOST = "localhost"
# The application user credentials created in your local Couchbase UI.
DB_USERNAME = "app_user"
DB_PASSWORD = "xxxxxx"
BUCKET_NAME = "travel-sample"
# The name of the FTS index we created in the local Couchbase UI.
INDEX_NAME = "hybrid-hotel-search-index"
# The name of the scope the index is on.
SCOPE_NAME = "inventory"
# The name of the collection we are targeting.
COLLECTION_NAME = "hotel"

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
    Performs a hybrid search using the FTS service on a specific scope.

    Args:
        cluster: The connected Couchbase Cluster object.
        query_text: The user's search query string.
        num_results: The number of results to return.
    """
    if not cluster:
        return

    print(f"\nPerforming hybrid search for: '{query_text}'...")

    try:
        # Get a handle to the specific scope and collection we want to search within.
        scope = cluster.bucket(BUCKET_NAME).scope(SCOPE_NAME)
        collection = scope.collection(COLLECTION_NAME)

        # Generate the vector for the user's query.
        print("Generating query vector...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        query_vector = model.encode(query_text).tolist()
        print("Vector generated.")

        # --- Define the Hybrid Search Query using the SDK's Builder Pattern ---

        # 1. Define the keyword search portion of the query.
        keyword_query = MatchQuery(query_text, field="description", boost=0.5)

        # 2. Define the vector search portion of the query.
        vector_query = VectorQuery('description_embedding', query_vector, num_candidates=num_results)

        # 3. Create a VectorSearch object, a container for one or more vector queries.
        vector_search = VectorSearch.from_vector_query(vector_query)

        # 4. Create a SearchRequest object containing both the keyword
        #    query and the vector search object.
        search_req = SearchRequest.create(keyword_query).with_vector_search(vector_search)

        # 5. Define SearchOptions to control the output.
        #    We ask for fields, but will have a fallback if they aren't stored in the index.
        search_options = SearchOptions(limit=num_results, fields=["name", "description"])

        # 6. Execute the search query on the scope object.
        print("Executing FTS hybrid search query on 'inventory' scope...")
        result = scope.search(
            INDEX_NAME,
            search_req,
            search_options
        )

        # --- Display the Results ---
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

        # Display performance metrics from the search.
        metrics = result.metadata().metrics()
        
        # ** THE FIX **: Convert the 'took' timedelta object to milliseconds before formatting.
        took_ms = metrics.took().total_seconds() * 1000
        print(f"\nQuery completed in {took_ms:.2f} ms")
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
