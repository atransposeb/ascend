from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parent / ".env"
print(f"ENV_PATH: {env_path}")
print(f"Exists: {env_path.exists()}")

load_dotenv(str(env_path))

print(f"GOOGLE_CLIENT_ID: {repr(os.getenv('GOOGLE_CLIENT_ID', ''))}")
print(f"GOOGLE_CLIENT_SECRET: {repr(os.getenv('GOOGLE_CLIENT_SECRET', ''))}")
