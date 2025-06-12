#
# Couchbase Technical Challenge - Data Enrichment Script (Local Docker Version)
#
# This script connects to a local Couchbase Docker instance, reads documents
# from the 'travel-sample' bucket, generates vector embeddings for hotel
# descriptions, and updates the documents with the new vector data.
#

import argparse
from datetime import timedelta

# SentenceTransformers is used to generate vector embeddings from text.
from sentence_transformers import SentenceTransformer

# Import necessary components from the Couchbase Python SDK.
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import CouchbaseException
# Import QueryOptions to parameterize our SQL++ query.
from couchbase.options import ClusterOptions, QueryOptions


# --- 1. Configuration (Local Docker) ---

# Update these placeholders with your local Couchbase Docker instance details.
DB_HOST = "localhost"
# The administrator credentials for your local Docker instance.
DB_USERNAME = "Administrator"
DB_PASSWORD = "xxxxxx"
# The name of the bucket containing the data.
BUCKET_NAME = "travel-sample"

# The embedding model to use.
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# The field in the document we will create embeddings from.
SOURCE_TEXT_FIELD = 'description'
# The new field where the vector embedding will be stored.
TARGET_EMBEDDING_FIELD = 'description_embedding'


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


def generate_and_store_embeddings():
    """
    Fetches hotel documents, generates embeddings for the description,
    and updates the documents in the local Couchbase instance.
    """
    cluster = connect_to_couchbase()
    if not cluster:
        return

    try:
        collection = cluster.bucket(BUCKET_NAME).scope("inventory").collection("hotel")

        # SQL++ query to retrieve all hotel document IDs
        query = f"SELECT RAW meta(h).id FROM `{BUCKET_NAME}`.inventory.hotel h"
        print(f"\nExecuting query to fetch hotel document IDs...")
        result = cluster.query(query)
        doc_ids = [row for row in result]
        print(f"Found {len(doc_ids)} hotel documents to process.")

        # Load the embedding model
        print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("Model loaded successfully.")

        # Process each document
        for i, doc_id in enumerate(doc_ids):
            # Fetch the full document
            doc = collection.get(doc_id).content_as[dict]

            # Get the text to be embedded, handle missing fields gracefully
            text_to_embed = doc.get(SOURCE_TEXT_FIELD)

            if text_to_embed and isinstance(text_to_embed, str):
                # Generate the embedding
                embedding = model.encode(text_to_embed).tolist()

                # Add the new embedding field to the document
                doc[TARGET_EMBEDDING_FIELD] = embedding

                # UPSERT the updated document back into Couchbase
                collection.upsert(doc_id, doc)
                print(f"({i+1}/{len(doc_ids)}) Successfully updated document: {doc_id}")
            else:
                print(f"({i+1}/{len(doc_ids)}) Skipped document (missing or invalid text field): {doc_id}")

        print("\nEmbedding generation and document updates complete.")

    except CouchbaseException as e:
        print(f"A Couchbase error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    generate_and_store_embeddings()

