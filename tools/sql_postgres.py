import os
import hashlib
import logging
from typing import Dict, Any, Tuple, List, Optional

from dotenv import load_dotenv
from soupsieve import match
from sympy import re
load_dotenv()

from sqlalchemy import create_engine, text, inspect
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _pg_uri() -> str:
    host = os.getenv("PG_HOST")
    port = int(os.getenv("PG_PORT"))
    db   = os.getenv("PG_DB")
    user = os.getenv("PG_USER")
    pwd  = os.getenv("PG_PASSWORD")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


# ============================================================================
# ORIGINAL IMPLEMENTATION (KEPT AS BACKUP)
# ============================================================================
# class SQLTool:
#     def __init__(self):
#         # Try OpenRouter first, fallback to OpenAI
#         api_key = os.getenv("OPEN_ROUTER_KEY")
#         if not api_key:
#             raise RuntimeError("Missing OPEN_ROUTER_KEY environment variable")
#
#         # Get model from environment or use default
#         llm_model = os.getenv("OPENROUTER_MODEL")
#         base_url = os.getenv("BASE_URL")
#         
#         self.engine = create_engine(_pg_uri(), pool_pre_ping=True)
#         self.db = SQLDatabase.from_uri(_pg_uri())
#         
#         # Use OpenRouter if OPEN_ROUTER_KEY is set
#         if os.getenv("OPEN_ROUTER_KEY"):
#             self.llm = ChatOpenAI(
#                 api_key=api_key,
#                 model=llm_model,
#                 temperature=0,
#                 base_url=base_url,
#                 model_kwargs={
#                     "response_format": {"type": "text"}
#                 }
#             )
#         else:
#             # Fallback to direct OpenAI
#             self.llm = ChatOpenAI(api_key=api_key, model=llm_model, temperature=0)
#
#         allowlist = os.getenv("SQL_ALLOWED_TABLES", "products")
#         self.allowed_tables = [t.strip().lower() for t in allowlist.split(",") if t.strip()]
#
#     def run(self, question: str) -> Dict[str, Any]:
#         chain = create_sql_query_chain(self.llm, self.db)
#         sql = chain.invoke({"question": question}).strip()
#         
#         # Simple cleanup for OpenRouter responses
#         # Remove common prefixes and markdown code blocks
#         if sql.startswith("SQLQuery:") or sql.startswith("SQL Query:"):
#             sql = sql.split(":", 1)[1].strip()
#         
#         # Remove markdown code blocks
#         sql = sql.replace("```sql", "").replace("```", "").strip()
#         
#         # If multi-line, join lines
#         if "\n" in sql:
#             sql = " ".join(line.strip() for line in sql.split("\n") if line.strip())
#
#         if self.allowed_tables:
#             lowered = sql.lower()
#             for t in self.allowed_tables:
#                 if f" {t} " in lowered or f" {t}(" in lowered or f" {t}\n" in lowered:
#                     break
#             else:
#                 return {"sql": sql, "rows": [], "note": "Blocked by table allowlist"}
#
#         with self.engine.connect() as conn:
#             try:
#                 result = conn.execute(text(sql))
#                 rows = [dict(r._mapping) for r in result]
#                 return {"sql": sql, "rows": rows}
#             except Exception as e:
#                 return {"sql": sql, "rows": [], "error": str(e)}


# ============================================================================
# ENHANCED IMPLEMENTATION WITH AUTOMATIC QUERY HANDLING
# ============================================================================
class SQLTool:
    """SQL Tool with automatic query handling capabilities:
    - Schema awareness and validation
    - Automatic error recovery
    - Intelligent result formatting
    - Security validation
    """
    
    def __init__(self):
        """Initialize  SQL tool with schema awareness"""
        # API Configuration
        api_key = os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise RuntimeError("Missing OPEN_ROUTER_KEY environment variable")

        llm_model = os.getenv("OPENROUTER_MODEL")
        base_url = os.getenv("BASE_URL")
        
        # Database setup
        self.engine = create_engine(_pg_uri(), pool_pre_ping=True)
        self.db = SQLDatabase.from_uri(_pg_uri())
        
        # LLM setup
        if os.getenv("OPEN_ROUTER_KEY"):
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=llm_model,
                temperature=0,
                base_url=base_url,
                model_kwargs={
                    "response_format": {"type": "text"}
                }
            )
        else:
            self.llm = ChatOpenAI(api_key=api_key, model=llm_model, temperature=0)

        # Table allowlist
        allowlist = os.getenv("SQL_ALLOWED_TABLES", "products")
        self.allowed_tables = [t.strip().lower() for t in allowlist.split(",") if t.strip()]
        
        # Load database schema
        self.schema_info = self._load_schema_info()
        
    def _load_schema_info(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Load database schema information including tables, columns, and types
        Returns:
            Dictionary mapping table names to their column information
        """
        schema = {}
        try:
            inspector = inspect(self.engine)
            
            # Get all table names
            table_names = inspector.get_table_names()
            
            for table_name in table_names:
                # Get columns for each table
                columns = inspector.get_columns(table_name)
                schema[table_name] = [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True)
                    }
                    for col in columns
                ]
            
            logger.info(f"📊 Loaded schema for tables: {', '.join(table_names)}")
            return schema
            
        except Exception as e:
            logger.error(f"❌ Failed to load schema: {e}")
            return {}
    
    def _build_schema_context(self) -> str:
        """
        Build a formatted schema description for the LLM
        Returns:
            Formatted string describing database schema
        """
        if not self.schema_info:
            return "No schema information available."
        
        schema_lines = []
        for table_name, columns in self.schema_info.items():
            col_list = ", ".join([f"{col['name']} ({col['type']})" for col in columns])
            schema_lines.append(f"Table '{table_name}': {col_list}")
        
        return "\n".join(schema_lines)
    
    def _validate_sql(self, sql: str) -> Tuple[bool, str]:
        """
        Validate SQL query for security and correctness
        
        Args:
            sql: SQL query string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        sql_lower = sql.lower().strip()
        
        # Check for dangerous operations
        dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
        for keyword in dangerous_keywords:
            if f" {keyword} " in f" {sql_lower} " or sql_lower.startswith(f"{keyword} "):
                return False, f"Dangerous operation '{keyword.upper()}' not allowed. Only SELECT queries are permitted."
        
        # Check for required SELECT
        if not sql_lower.startswith('select'):
            return False, "Only SELECT queries are allowed"
        
        # Check if query references known tables (if schema loaded)
        if self.schema_info:
            found_table = False
            for table in self.schema_info.keys():
                if table.lower() in sql_lower:
                    found_table = True
                    break
            
            if not found_table:
                return False, f"Query must reference known tables: {', '.join(self.schema_info.keys())}"
        
        # Check table allowlist if configured
        if self.allowed_tables:
            found_allowed = False
            for t in self.allowed_tables:
                if f" {t} " in f" {sql_lower} " or f"{t}(" in sql_lower:
                    found_allowed = True
                    break
            
            if not found_allowed:
                return False, f"Query blocked by table allowlist. Allowed tables: {', '.join(self.allowed_tables)}"
        
        return True, "Valid"
    
    def _clean_sql_response(self, sql: str) -> str:
        """
        Clean up SQL query from LLM response
        
        Args:
            sql: Raw SQL string from LLM
            
        Returns:
            Cleaned SQL query
        """
        sql = sql.strip()
       
        prefixes = ["SQLQuery:", "SQL Query:", "Query:", "SQL:"]
        for prefix in prefixes:
            if sql.startswith(prefix):
                sql = sql[len(prefix):].strip()

        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        if "\n" in sql:
            sql = " ".join(line.strip() for line in sql.split("\n") if line.strip())
        
        return sql
    
    
    def _generate_sql(self, question: str) -> str:
        """
        Generate SQL query from natural language question
        
        Args:
            question: Natural language question
            
        Returns:
            Generated SQL query
        """
    
        schema_context = self._build_schema_context()
        
        enhanced_question = f"""Database Schema:
            {schema_context}

            User Question: {question}

            Generate a SQL query that:
            1. Only uses columns that exist in the schema above
            2. Uses proper JOINs if multiple tables are needed
            3. Includes appropriate WHERE conditions
            4. Uses LIMIT clause to prevent excessive results (default LIMIT 100)
            5. Returns all relevant columns for the user's question
            """
        
        # Generate SQL using LangChain
        chain = create_sql_query_chain(self.llm, self.db)
        sql = chain.invoke({"question": enhanced_question}).strip()
        import re

        match = re.search(r'SQLQuery:\s*(.*)', sql, re.DOTALL)
        if match:
            sql_query = match.group(1).strip()
        sql = self._clean_sql_response(sql_query)
        
        
        return sql
    
    def _execute_with_retry(self, sql: str, question: str, retry_count: int = 0) -> Dict[str, Any]:
        """
        Execute SQL with automatic retry on failure
        
        Args:
            sql: SQL query to execute
            question: Original question (for retry)
            retry_count: Current retry attempt
            
        Returns:
            Query results dictionary
        """
        with self.engine.connect() as conn:
            try:
                result = conn.execute(text(sql))
                rows = [dict(r._mapping) for r in result]
                
                # Format results intelligently
                total_count = len(rows)
                
                if total_count > 100:
                    return {
                        "sql": sql,
                        "rows": rows[:100],
                        "total_count": total_count,
                        "truncated": True,
                        "message": f"Showing first 100 of {total_count} results. Query was successful."
                    }
                
                return {
                    "sql": sql,
                    "rows": rows,
                    "total_count": total_count,
                    "truncated": False
                }
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ SQL execution error: {error_msg}")
                
                # Attempt automatic retry with error correction (max 2 retries)
                if retry_count < 2:
                    logger.info(f"🔄 Attempting to fix query (attempt {retry_count + 1}/2)...")
                    
                    try:
                        # Ask LLM to fix the SQL based on error
                        fix_prompt = f"""The following SQL query failed with an error:
                                    SQL Query:
                                    {sql}
                                    Error Message:
                                    {error_msg}
                                    Database Schema:
                                    {self._build_schema_context()}
                                    Please generate a corrected SQL query that fixes this error. Consider:
                                    1. Check if column names exist in the schema
                                    2. Fix any syntax errors
                                    3. Ensure proper table references
                                    4. Add necessary JOINs if accessing multiple tables
                                    Generate only the corrected SQL query."""
                        
                        chain = create_sql_query_chain(self.llm, self.db)
                        fixed_sql = chain.invoke({"question": fix_prompt}).strip()
                        fixed_sql = self._clean_sql_response(fixed_sql)
                        
                        # Validate the fixed SQL
                        is_valid, validation_msg = self._validate_sql(fixed_sql)
                        if not is_valid:
                            return {
                                "sql": sql,
                                "rows": [],
                                "error": error_msg,
                                "fix_attempted": True,
                                "fix_validation_error": validation_msg
                            }
                        
                        # Retry with fixed SQL
                        logger.info(f"🔧 Retrying with fixed SQL: {fixed_sql[:100]}...")
                        return self._execute_with_retry(fixed_sql, question, retry_count + 1)
                        
                    except Exception as fix_error:
                        logger.error(f"❌ Failed to fix query: {fix_error}")
                
                return {
                    "sql": sql,
                    "rows": [],
                    "error": error_msg,
                    "retry_count": retry_count
                }
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        Execute natural language query against database with automatic handling
        
        Args:
            question: Natural language question about the database
            
        Returns:
            Dictionary containing:
                - sql: Generated SQL query
                - rows: Query results (list of dicts)
                - total_count: Number of results
                - truncated: Whether results were truncated
                - error: Error message if query failed
                - Additional metadata
        """
        try:
            
            # Generate SQL query
            sql = self._generate_sql(question)
            
            # Validate SQL query
            is_valid, validation_msg = self._validate_sql(sql)
            if not is_valid:
                logger.warning(f"⚠️ Validation failed: {validation_msg}")
                return {
                    "sql": sql,
                    "rows": [],
                    "error": f"Query validation failed: {validation_msg}"
                }
            
            # Execute query with retry capability
            result = self._execute_with_retry(sql, question)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in run(): {e}")
            return {
                "sql": "",
                "rows": [],
                "error": f"Unexpected error: {str(e)}"
            }

    
    # def get_schema(self) -> Dict[str, List[Dict[str, str]]]:
    #     """
    #     Get the loaded database schema
        
    #     Returns:
    #         Database schema information
    #     """
    #     return self.schema_info
