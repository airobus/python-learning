from langchain_community.vectorstores import SupabaseVectorStore
import os
from dotenv import load_dotenv
from supabase.client import Client, create_client
from .robus_embedding import embeddings

load_dotenv(override=True)

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase_token = os.environ.get("SUPABASE_TOKEN")

print(f"supabase_url: " + supabase_url)
print(f"supabase_token: " + supabase_token)
print(f"supabase_key: " + supabase_key)

supabase: Client = create_client(supabase_url, supabase_key)

vectorstore_small = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="bge_small_vector",
    query_name="bge_small_match_documents",
)
