import sqlite3
from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel
from typing import List
from langchain_core.utils.function_calling import convert_to_openai_tool

# Global connection variable
conn = None

# Establishing a connection to the SQLite database
def initialize_connection(file):
    """Initialize the global database connection."""
    global conn
    if conn is None:
        conn = sqlite3.connect(f'file:{file.name}?mode=rw', uri=True)

def list_tables():
    """List all tables in the SQLite database."""
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)

class RunSQLiteQueryArgSchema(BaseModel):
    query: str

def run_sqlite_query(query):
    """Run a given SQLite query and return the results."""
    c = conn.cursor()
    c.execute(query)
    return c.fetchall()

run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a SQLite query",
    func=run_sqlite_query,
    args_schema=RunSQLiteQueryArgSchema
)

class DescribeTablesArgSchema(BaseModel):
    table_names: List[str]

def describe_tables(table_names):
    """Given a list of table names, return the schema of those tables."""
    c = conn.cursor()
    tables = ', '.join("'" + table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables});")
    return "\n".join(row[0] for row in rows if row[0] is not None)

describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, returns the schema of those tables",
    func=describe_tables,
    args_schema=DescribeTablesArgSchema
)