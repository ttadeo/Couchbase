# Couchbase
This repository contains artifacts for the Couchbase Technical Challenge
# Couchbase & LangChain: A Hybrid Search CLI

This project is a Python-based command-line application that fulfills the Couchbase Technical Challenge by demonstrating a powerful hybrid search system using Couchbase Capella. It leverages the `travel-sample` dataset to provide a "semantic cache" capable of understanding user queries based on both literal keywords and semantic meaning.

The application was successfully developed, tested, and deployed against a Couchbase Capella Free Tier environment after a thorough troubleshooting and diagnostic process.

## Features

-   **Data Enrichment:** Programmatically generates vector embeddings for hotel descriptions using the `sentence-transformers` library.
-   **Hybrid Search:** Combines traditional keyword (Full-Text Search) and modern semantic (vector) search in a single query to deliver highly relevant results.
-   **Command-Line Interface:** Provides a simple and effective CLI for users to interact with the search system.
-   **Robust Architecture:** Built using the modern Couchbase Python SDK (v4.x) and demonstrates best practices for connecting to and querying Couchbase services.

## Architecture

The project consists of two main Python scripts and a Couchbase Capella database:

1.  **`generate_embeddings.py`**: This script connects to the Capella cluster, reads all documents from the `travel-sample.inventory.hotel` collection, generates a 384-dimensional vector embedding from each hotel's `description` field, and updates the document in place by adding a new `description_embedding` field.

2.  **`semantic_cache_cli.py`**: This is the user-facing application. It takes a natural language query from the command line, generates a corresponding query vector, and executes a hybrid search against a specialized Couchbase Full-Text Search (FTS) index using the Python SDK.

3.  **Couchbase FTS Index**: The backend is powered by a custom FTS index (`capella-hybrid-hotel-index`) that is configured with two key mappings:
    -   A **text** mapping on the `description` field for efficient keyword matching.
    -   A **vector** mapping on the `description_embedding` field for high-speed cosine similarity searches.

## Setup and Usage on Capella

### Step 1: Configure Capella Instance

1.  Log in to your Capella cluster and ensure the `travel-sample` bucket is loaded.
2.  Navigate to **Settings -> Cluster Access** and create a new database user (e.g., `capella_app_user`). Grant this user the following roles on the `travel-sample` bucket:
    -   `Bucket Admin`
    -   `Search Admin`
    -   `Data Reader`
    -   `Query Select`
3.  Navigate to the **Search** tab and create a new FTS index (e.g., `capella-hybrid-hotel-index`) on the `travel-sample.inventory.hotel` collection, with child mappings for `description` (text) and `description_embedding` (vector).

### Step 2: Prepare Python Environment

1.  Clone this repository.
2.  Install the required Python packages:
    ```bash
    pip install couchbase sentence-transformers
    ```
3.  Update the configuration variables in both Python scripts with your Capella endpoint and the new user's credentials.

### Step 3: Generate Embeddings

Run the enrichment script once to populate your Capella database with the vector data.

```bash
python generate_embeddings.py

Step 4: Run a Hybrid Search
You can now use the command-line application to perform searches.

# Example command
python semantic_cache_cli.py --query "a charming hotel with a quiet garden"
