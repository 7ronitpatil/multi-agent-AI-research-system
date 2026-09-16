import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver


load_dotenv()


DB_URI = os.getenv("DATABASE_URL")


def setup_database():
    print("Setting up PostgreSQL checkpoint database...")

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()

    print("Database setup completed.")


if __name__ == "__main__":
    setup_database()