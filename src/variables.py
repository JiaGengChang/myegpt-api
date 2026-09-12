import os

DBHOSTNAME=os.environ.get("DBHOSTNAME")
DBUSERNAME=os.environ.get("DBUSERNAME")
DBPASSWORD=os.environ.get("DBPASSWORD")
MODEL_ID=os.environ.get("MODEL_ID")
EVAL_MODEL_ID=os.environ.get("EVAL_MODEL_ID")
SERVER_BASE_URL=os.environ.get("SERVER_BASE_URL")
EMBEDDINGS_MODEL_PROVIDER = os.environ.get("EMBEDDINGS_MODEL_PROVIDER")
EMBEDDINGS_TABLE_SUFFIX = os.environ.get("EMBEDDINGS_TABLE_SUFFIX")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT")

assert DBHOSTNAME is not None, "DBHOSTNAME environment variable is not set"
assert DBUSERNAME is not None, "DBUSERNAME environment variable is not set"
assert DBPASSWORD is not None, "DBPASSWORD environment variable is not set"
assert MODEL_ID is not None, "MODEL_ID environment variable is not set"
assert SERVER_BASE_URL is not None, "SERVER_BASE_URL environment variable is not set"
assert EMBEDDINGS_MODEL_PROVIDER is not None, "EMBEDDINGS_MODEL_PROVIDER environment variable is not set"
assert EMBEDDINGS_TABLE_SUFFIX is not None, "EMBEDDINGS_TABLE_SUFFIX environment variable is not set"
assert LANGSMITH_PROJECT is not None, "LANGSMITH_PROJECT environment variable is not set"

# derived variables
COMMPASS_DSN=f"dbname=commpass user={DBUSERNAME} password={DBPASSWORD} host={DBHOSTNAME} port=5432"
COMMPASS_AUTH_DSN=f"dbname=commpass user={DBUSERNAME} password={DBPASSWORD} host={DBHOSTNAME} options='-c search_path=auth'"
COMMPASS_DB_URI=f"postgresql+psycopg://{DBUSERNAME}:{DBPASSWORD}@{DBHOSTNAME}/commpass"
COMMPASS_DB_URI_POSTGRES=f"postgresql+psycopg://{DBUSERNAME}:{DBPASSWORD}@{DBHOSTNAME}/commpass"
COMMPASS_MEMORY_DB_URI=f"postgresql://{DBUSERNAME}:{DBPASSWORD}@{DBHOSTNAME}:5432/commpass?options=-csearch_path%3dcheckpoints"

__all__ = [
    "API_BYPASS_TOKEN",
    "COMMPASS_AUTH_DSN",
    "COMMPASS_DB_URI",
    "COMMPASS_DB_URI_POSTGRES",
    "COMMPASS_DSN",
    "COMMPASS_MEMORY_DB_URI",
    "EMBEDDINGS_MODEL_PROVIDER",
    "EMBEDDINGS_TABLE_SUFFIX",
    "MODEL_ID",
    "EVAL_MODEL_ID",
    "SERVER_BASE_URL",
]