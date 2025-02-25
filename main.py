# # # # # # # from fastapi import FastAPI, Depends
# # # # # # # import asyncpg
# # # # # # # from google.generativeai import GenerativeModel
# # # # # # # from database import get_review_pool

# # # # # # # # Initialize FastAPI app
# # # # # # # app = FastAPI()


# # # # # # # db_pool = None

# # # # # # # # Function to create database pool
# # # # # # # async def create_db_pool():
# # # # # # #     global db_pool
# # # # # # #     db_pool = await asyncpg.create_pool(dsn="postgresql://postgres:prodloop%40gtm@34.93.88.145:5432/postgres")

# # # # # # # # FastAPI startup event to initialize DB pool
# # # # # # # @app.on_event("startup")
# # # # # # # async def startup():
# # # # # # #     await create_db_pool()

# # # # # # # # FastAPI shutdown event to close DB pool
# # # # # # # @app.on_event("shutdown")
# # # # # # # async def shutdown():
# # # # # # #     await db_pool.close()

# # # # # # # # Dependency to get the database pool
# # # # # # # async def get_db_pool():
# # # # # # #     if db_pool is None:
# # # # # # #         raise RuntimeError("Database pool is not initialized")
# # # # # # #     return db_pool
# # # # # # # # Database connection pool (assuming it's already set up)
# # # # # # # # async def get_db_pool():
# # # # # # # #     return get_review_pool()  # Ensure `get_status_poo` is properly initialized elsewhere

# # # # # # # # Function to fetch relevant documents (assuming text search, not vectors)
# # # # # # # async def fetch_relevant_docs(query: str, pool):
# # # # # # #     async with pool.acquire() as conn:
# # # # # # #         rows = await conn.fetch(
# # # # # # #             """
# # # # # # #             SELECT content FROM documents
# # # # # # #             WHERE content ILIKE $1
# # # # # # #             LIMIT 5;
# # # # # # #             """,
# # # # # # #             f"%{query}%"
# # # # # # #         )
# # # # # # #         return [row["content"] for row in rows]

# # # # # # # # Function to generate an answer using RAG
# # # # # # # model = GenerativeModel("gemini-pro")
# # # # # # # async def generate_answer(query: str, pool):
# # # # # # #     docs = await fetch_relevant_docs(query, pool)
    
# # # # # # #     context = "\n".join(docs)
# # # # # # #     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
# # # # # # #     response = model.generate_content(prompt)
# # # # # # #     return response.text

# # # # # # # # API endpoint for RAG-based retrieval and generation
# # # # # # # @app.get("/rag/")
# # # # # # # async def rag_search(query: str, pool=Depends(get_db_pool)):
# # # # # # #     response = await generate_answer(query, pool)
# # # # # # #     return {"answer": response}

# # # # # # import vanna
# # # # # # import asyncpg
# # # # # # import os
# # # # # # from database import get_review_pool

# # # # # # # Initialize Vanna AI
# # # # # # vn = vanna.Vanna(model="postgres")  # Specify that we're using PostgreSQL

# # # # # # # Set database credentials
# # # # # # DB_CONFIG = {
# # # # # #     "user": "your_username",
# # # # # #     "password": "your_password",
# # # # # #     "database": "your_database",
# # # # # #     "host": "your_host",
# # # # # #     "port": 5432  # Default PostgreSQL port
# # # # # # }

# # # # # # # Create connection pool
# # # # # # async def get_db_pool():
# # # # # #     return await asyncpg.create_pool(**DB_CONFIG)

# # # # # # # Function to fetch relevant documents using Vanna AI
# # # # # # async def fetch_relevant_docs(query: str, pool):
# # # # # #     async with pool.acquire() as conn:
# # # # # #         # Use Vanna AI to generate a SQL query for relevant docs
# # # # # #         sql_query = vn.generate_sql(query)
        
# # # # # #         if not sql_query:
# # # # # #             return ["No relevant documents found."]

# # # # # #         # Execute the generated query
# # # # # #         rows = await conn.fetch(sql_query)
# # # # # #         return [row["content"] for row in rows]

# # # # # # # Function to generate an answer using RAG
# # # # # # async def generate_answer(query: str, pool):
# # # # # #     docs = await fetch_relevant_docs(query, pool)
# # # # # #     context = "\n".join(docs)
# # # # # #     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

# # # # # #     response = vn.ask(prompt)  # Use Vanna AI for answering
# # # # # #     return response

# # # # # from fastapi import FastAPI, Depends
# # # # # from google.generativeai import GenerativeModel
# # # # # from vanna.remote import Vanna
# # # # # import asyncpg
# # # # # from database import get_status_pool

# # # # # # Initialize FastAPI app
# # # # # app = FastAPI()

# # # # # # Initialize Vanna AI
# # # # # vn = Vanna(model="postgres")  # Using PostgreSQL for SQL generation

# # # # # # Your existing database connection pool
# # # # # async def get_db_pool():
# # # # #     return get_status_pool  # Ensure `get_status_pool` is initialized elsewhere

# # # # # # Function to fetch relevant documents using Vanna AI
# # # # # async def fetch_relevant_docs(query: str, pool):
# # # # #     async with pool.acquire() as conn:
# # # # #         # Generate a SQL query using Vanna AI
# # # # #         sql_query = vn.generate_sql(query)
        
# # # # #         if not sql_query:
# # # # #             return ["No relevant documents found."]
        
# # # # #         try:
# # # # #             rows = await conn.fetch(sql_query)
# # # # #             return [row["content"] for row in rows]
# # # # #         except Exception as e:
# # # # #             return [f"Error executing query: {str(e)}"]

# # # # # # Function to generate an answer using RAG
# # # # # model = GenerativeModel("gemini-pro")
# # # # # async def generate_answer(query: str, pool):
# # # # #     docs = await fetch_relevant_docs(query, pool)
    
# # # # #     context = "\n".join(docs)
# # # # #     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
# # # # #     response = model.generate_content(prompt)
# # # # #     return response.text

# # # # # # API endpoint for RAG-based retrieval and generation
# # # # # @app.get("/rag/")
# # # # # async def rag_search(query: str, pool=Depends(get_db_pool)):
# # # # #     response = await generate_answer(query, pool)
# # # # #     return {"answer": response}

# # # # from fastapi import FastAPI, Depends
# # # # from vanna.remote import VannaDefault
# # # # import asyncpg
# # # # import vanna
# # # # from database import get_status_pool

# # # # # Initialize FastAPI app
# # # # app = FastAPI()

# # # # # Initialize Vanna
# # # # api_key = vanna.get_api_key('anirudhdevs91@gmail.com')  # Replace with your email
# # # # vn = VannaDefault(model='your_model_name', api_key=api_key)  # Replace with your model name

# # # # # Database connection pool
# # # # async def get_db_pool():
# # # #     # Ensure `get_status_pool` is properly initialized elsewhere
# # # #     pool = await get_status_pool()
# # # #     return pool

# # # # # Function to fetch relevant documents
# # # # async def fetch_relevant_docs(query: str, pool):
# # # #     async with pool.acquire() as conn:
# # # #         rows = await conn.fetch(
# # # #             """
# # # #             SELECT content FROM documents
# # # #             WHERE content ILIKE $1
# # # #             LIMIT 5;
# # # #             """,
# # # #             f"%{query}%"
# # # #         )
# # # #         return [row["content"] for row in rows]

# # # # # Function to generate an answer using Vanna
# # # # async def generate_answer(query: str, pool):
# # # #     docs = await fetch_relevant_docs(query, pool)
# # # #     context = "\n".join(docs)
# # # #     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
# # # #     response = vn.ask(prompt)
# # # #     return response

# # # # # API endpoint for RAG-based retrieval and generation
# # # # @app.get("/rag/")
# # # # async def rag_search(query: str, pool=Depends(get_db_pool)):
# # # #     response = await generate_answer(query, pool)
# # # #     return {"answer": response}

# # # import os
# # # import asyncpg
# # # import numpy as np
# # # from fastapi import FastAPI, Depends
# # # from pydantic import BaseModel
# # # import vanna
# # # from database import get_status_pool

# # # app = FastAPI()

# # # # api_key = vanna.get_api_key('anirudhdevs91@gmail.com')  # Replace with your email
# # # # vn = VannaDefault(model='your_model_name', api_key=api_key)  # Replace with your model name

# # # # Load Vanna AI
# # # vn = vanna.VannaDefault()
# # # vn.connect()


# # # async def get_db_pool():
# # #     # Ensure `get_status_pool` is properly initialized elsewhere
# # #     pool = await get_status_pool()
# # #     return pool


# # # # Pydantic model for user query
# # # class QueryRequest(BaseModel):
# # #     query: str

# # # # Natural Language to SQL Query
# # # @app.post("/query/")
# # # async def natural_language_query(request: QueryRequest):
# # #     """ Convert natural language to SQL and fetch results from DB. """
# # #     sql_query = vn.ask(request.query)  # Vanna converts query to SQL
# # #     async with pool.acquire() as conn:
# # #         rows = await conn.fetch(sql_query)
# # #     return {"query": sql_query, "results": [dict(row) for row in rows]}

# # # # Vector Search for RAG
# # # @app.post("/rag/")
# # # async def rag_search(request: QueryRequest):
# # #     """ Perform vector similarity search for RAG. """
# # #     query_embedding = np.random.rand(1536).tolist()  # Replace with actual embedding logic
# # #     sql = """
# # #         SELECT content FROM document_vectors
# # #         ORDER BY embedding <-> $1
# # #         LIMIT 5
# # #     """
# # #     async with db_pool.acquire() as conn:
# # #         rows = await conn.fetch(sql, query_embedding)
# # #     return {"results": [dict(row) for row in rows]}

# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="0.0.0.0", port=8000)

# # import os
# # import asyncpg
# # import numpy as np
# # from fastapi import FastAPI, Depends,HTTPException
# # from pydantic import BaseModel
# # from vanna.remote import VannaDefault
# # from database import get_status_pool
# # import vanna


# # app = FastAPI()

# # # Load Vanna AI
# # # vn = vanna.VannaDefault()
# # # vn.connect()

# # api_key = vanna.get_api_key('anirudhdevs91@gmail.com')  # Replace with your email
# # vn = VannaDefault(model='your_model_name', api_key=api_key)  # Replace with your model name
# # async def get_db_pool():
# #     """Get a connection pool for the database."""
# #     return await get_status_pool()  # Ensure this is correctly implemented

# # # Pydantic model for user query
# # class QueryRequest(BaseModel):
# #     query: str

# # # Natural Language to SQL Query
# # # @app.post("/query/")
# # # async def natural_language_query(request: QueryRequest, pool: asyncpg.Pool = Depends(get_db_pool)):
# # #     """Convert natural language to SQL and fetch results from DB."""
# # #     sql_query = vn.ask(request.query)  # Vanna converts query to SQL
# # #     async with pool.acquire() as conn:
# # #         rows = await conn.fetch(sql_query)
# # #     return {"query": sql_query, "results": [dict(row) for row in rows]}

# # @app.post("/query/")
# # async def natural_language_query(data: dict,pool: asyncpg.Pool = Depends(get_db_pool)):
# #     """ Convert natural language to SQL and fetch results from DB. """
# #     query = data.get("query", "")
# #     if not query:
# #         raise HTTPException(status_code=400, detail="Query is required")

# #     sql_query = vn.ask(query)
# #     print(f"Generated SQL query: {sql_query}")
    
# #     if isinstance(sql_query, tuple):  # If it's a tuple, extract the SQL string
# #         sql_query = sql_query[0]

# #     async with pool.acquire() as conn:
# #         rows = await conn.fetch(sql_query)
    
# #     return {"query": sql_query, "results": [dict(row) for row in rows]}
# # # Vector Search for RAG
# # @app.post("/rag/")
# # async def rag_search(request: QueryRequest, pool: asyncpg.Pool = Depends(get_db_pool)):
# #     """Perform vector similarity search for RAG."""
# #     query_embedding = np.random.rand(1536).tolist()  # Replace with real embedding logic
# #     sql = """
# #         SELECT content FROM document_vectors
# #         ORDER BY embedding <-> $1
# #         LIMIT 5
# #     """
# #     async with pool.acquire() as conn:
# #         rows = await conn.fetch(sql, query_embedding)
# #     return {"results": [dict(row) for row in rows]}

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)

# # import vanna as vn
# # import asyncio
# # import asyncpg  # Assuming you're using asyncpg for PostgreSQL
# # from database import get_review_pool
# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel
# # from contextlib import asynccontextmanager

# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     global pool
# #     # Initialize the connection pool when the app starts
# #     pool = await get_status_pool()
# #     yield
# #     # Close the connection pool when the app shuts down
# #     await pool.close()

# # # Replace with your actual Vanna API key
# # # vn.set_api_key('be54b4c580b64e16aa77e31594a2e09b')
# # from vanna.remote import VannaDefault
# # app = FastAPI(lifespan=lifespan)
# # api_key = 'be54b4c580b64e16aa77e31594a2e09b'
# # vanna_model_name = 'prodloop'
                    
# # vn = VannaDefault(model=vanna_model_name, api_key=api_key)

# # # Replace with your actual database connection pool setup
# # async def get_status_pool():
# #     pool = await get_review_pool()
# #     return pool


# # class QuestionRequest(BaseModel):
# #     question: str


# # # Function to fetch the database schema (DDL) automatically
# # async def fetch_schema(pool):
# #     async with pool.acquire() as connection:
# #         # Fetch all table names in the public schema
# #         tables = await connection.fetch(
# #             "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
# #         )
# #         schema_ddl = []
# #         for table in tables:
# #             table_name = table['table_name']
# #             # Fetch the columns and their definitions for each table
# #             columns = await connection.fetch(
# #                 f"""
# #                 SELECT column_name, data_type, is_nullable, column_default
# #                 FROM information_schema.columns
# #                 WHERE table_name = $1;
# #                 """,
# #                 table_name
# #             )
# #             # Generate the DDL for the table
# #             ddl = f"CREATE TABLE {table_name} (\n"
# #             ddl += ",\n".join(
# #                 f"    {col['column_name']} {col['data_type']} "
# #                 f"{'NOT NULL' if col['is_nullable'] == 'NO' else ''} "
# #                 f"{'DEFAULT ' + col['column_default'] if col['column_default'] else ''}"
# #                 for col in columns
# #             )
# #             ddl += "\n);"
# #             schema_ddl.append(ddl)
# #         return "\n".join(schema_ddl)

# # # Function to train Vanna AI on the schema
# # async def train_vanna(pool):
# #     schema_ddl = await fetch_schema(pool)
# #     vn.train(ddl=schema_ddl)
# #     print("Vanna AI has been trained on the database schema.")

# # # Function to generate SQL and execute it using the connection pool
# # async def ask_question(pool, question):
# #     # Generate SQL from the natural language question
# #     sql_query = vn.generate_sql(question)
# #     print(f"Generated SQL: {sql_query}")

# #     # Execute the SQL query using the connection pool
# #     async with pool.acquire() as connection:
# #         result = await connection.fetch(sql_query)
# #         return result

# # # FastAPI endpoint to ask a question
# # @app.post("/ask")
# # async def ask(request: QuestionRequest):
# #     try:
# #         # Get the database connection pool
# #         pool = await get_status_pool()

# #         # Train Vanna AI on the database schema (if not already trained)
# #         await train_vanna(pool)

# #         # Ask the question
# #         result = await ask_question(pool, request.question)

# #         # Close the connection pool
# #         await pool.close()

# #         # Return the result
# #         return {"result": result}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# # # Run the FastAPI app
# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)
# # # # Function to fetch the database schema (DDL) automatically
# # async def fetch_schema(pool):
# #     async with pool.acquire() as connection:
# #         # Fetch all table names
# #         tables = await connection.fetch(
# #             "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
# #         )
# #         schema_ddl = []
# #         for table in tables:
# #             table_name = table['table_name']
# #             # Fetch the DDL for each table
# #             ddl = await connection.fetchval(
# #                 f"SELECT pg_get_tabledef('{table_name}');"
# #             )
# #             schema_ddl.append(ddl)
# #         return "\n".join(schema_ddl)

# # # Function to train Vanna AI on the schema
# # async def train_vanna(pool):
# #     schema_ddl = await fetch_schema(pool)
# #     vn.train(ddl=schema_ddl)
# #     print("Vanna AI has been trained on the database schema.")

# # # Function to generate SQL and execute it using the connection pool
# # async def ask_question(pool, question):
# #     # Generate SQL from the natural language question
# #     sql_query = vn.generate_sql(question)
# #     print(f"Generated SQL: {sql_query}")

# #     # Execute the SQL query using the connection pool
# #     async with pool.acquire() as connection:
# #         result = await connection.fetch(sql_query)
# #         return result

# # # Main function to run the workflow
# # async def main():
# #     # Get the database connection pool
# #     pool = await get_status_pool()

# #     # Train Vanna AI on the database schema
# #     await train_vanna(pool)

# #     # Ask a natural language question
# #     question = "What is the reviews of zomato?"
# #     result = await ask_question(pool, question)

# #     # Print the result
# #     print(f"Result: {result}")

# #     # Close the connection pool
# #     await pool.close()

# # # Run the main function
# # if __name__ == "__main__":
# #     asyncio.run(main())


# import vanna as vn
# import asyncpg
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from contextlib import asynccontextmanager
# from vanna.remote import VannaDefault
# from database import get_review_pool
# api_key = 'be54b4c580b64e16aa77e31594a2e09b'
# vanna_model_name = 'prodloop'
                    
# vn = VannaDefault(model=vanna_model_name, api_key=api_key)
# vn.allow_llm_to_see_data = True  # Add this line

# # Replace with your actual database connection pool setup
# async def get_status_pool():
#     pool = await get_review_pool()
#     return pool

# # Replace with your actual Vanna API key
# # vn.set_api_key('your_vanna_api_key')

# # Pydantic model for the request body
# class QuestionRequest(BaseModel):
#     question: str

# # Global connection pool
# pool = None

# # FastAPI lifespan handler to manage the connection pool
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global pool
#     # Initialize the connection pool when the app starts
#     pool = await get_status_pool()
#     yield
#     # Close the connection pool when the app shuts down
#     await pool.close()

# # FastAPI app with lifespan
# app = FastAPI(lifespan=lifespan)

# # Replace with your actual database connection pool setup
# async def get_status_pool():
#     return await asyncpg.create_pool(
#         host='34.93.88.145',
#         database='postgres',
#         user='postgres',
#         password='prodloop@gtm',
#         port=5432
#     )

# # Function to fetch the database schema (DDL) automatically
# async def fetch_schema(pool):
#     async with pool.acquire() as connection:
#         # Fetch all table names in the public schema
#         tables = await connection.fetch(
#             "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
#         )
#         schema_ddl = []
#         for table in tables:
#             table_name = table['table_name']
#             # Fetch the columns and their definitions for each table
#             columns = await connection.fetch(
#                 f"""
#                 SELECT column_name, data_type, is_nullable, column_default
#                 FROM information_schema.columns
#                 WHERE table_name = $1;
#                 """,
#                 table_name
#             )
#             # Generate the DDL for the table
#             ddl = f"CREATE TABLE {table_name} (\n"
#             ddl += ",\n".join(
#                 f"    {col['column_name']} {col['data_type']} "
#                 f"{'NOT NULL' if col['is_nullable'] == 'NO' else ''} "
#                 f"{'DEFAULT ' + col['column_default'] if col['column_default'] else ''}"
#                 for col in columns
#             )
#             ddl += "\n);"
#             schema_ddl.append(ddl)
#         return "\n".join(schema_ddl)

# # Function to train Vanna AI on the schema
# async def train_vanna(pool):
#     schema_ddl = await fetch_schema(pool)
#     vn.train(ddl=schema_ddl)
#     print("Vanna AI has been trained on the database schema.")

# # Function to generate SQL and execute it using the connection pool
# async def ask_question(pool, question):
#     # Generate SQL from the natural language question
#     sql_query = vn.generate_sql(question)
#     print(f"Generated SQL: {sql_query}")

#     # Execute the SQL query using the connection pool
#     async with pool.acquire() as connection:
#         result = await connection.fetch(sql_query)
#         return result

# # FastAPI endpoint to ask a question
# @app.post("/ask")
# async def ask(request: QuestionRequest):
#     try:
#         # Train Vanna AI on the database schema (if not already trained)
#         await train_vanna(pool)

#         # Ask the question
#         result = await ask_question(pool, request.question)

#         # Return the result
#         return {"result": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Run the FastAPI app
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import vanna as vn
import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from vanna.remote import VannaDefault
from database import get_review_pool  # Ensure this is correctly imported

# Vanna AI setup
api_key = 'be54b4c580b64e16aa77e31594a2e09b'
vanna_model_name = 'prodloop'
vn = VannaDefault(model=vanna_model_name, api_key=api_key)
# vn.allow_llm_to_see_data = True  # Explicitly allow LLM to see data

# Pydantic model for the request body
class QuestionRequest(BaseModel):
    question: str

# Global connection pool
pool = None

# FastAPI lifespan handler to manage the connection pool
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    # Initialize the connection pool when the app starts
    pool = await get_status_pool()
    yield
    # Close the connection pool when the app shuts down
    await pool.close()

# FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Replace with your actual database connection pool setup
async def get_status_pool():
    return await get_review_pool()  # Ensure this function returns a valid connection pool

# Function to fetch the database schema (DDL) automatically
async def fetch_schema(pool):
    async with pool.acquire() as connection:
        # Fetch all table names in the public schema
        tables = await connection.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        )
        schema_ddl = []
        for table in tables:
            table_name = table['table_name']
            # Fetch the columns and their definitions for each table
            columns = await connection.fetch(
                f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = $1;
                """,
                table_name
            )
            # Generate the DDL for the table
            ddl = f"CREATE TABLE {table_name} (\n"
            ddl += ",\n".join(
                f"    {col['column_name']} {col['data_type']} "
                f"{'NOT NULL' if col['is_nullable'] == 'NO' else ''} "
                f"{'DEFAULT ' + col['column_default'] if col['column_default'] else ''}"
                for col in columns
            )
            ddl += "\n);"
            schema_ddl.append(ddl)
        return "\n".join(schema_ddl)

# Function to train Vanna AI on the schema
async def train_vanna(pool):
    schema_ddl = await fetch_schema(pool)
    vn.train(ddl=schema_ddl)
    print("Vanna AI has been trained on the database schema.")

# Function to generate SQL and execute it using the connection pool
async def ask_question(pool, question):
    # Generate SQL from the natural language question
    sql_query = vn.generate_sql(question)
    print(f"Generated SQL: {sql_query}")

    # Execute the SQL query using the connection pool
    async with pool.acquire() as connection:
        result = await connection.fetch(sql_query)
        return result

# FastAPI endpoint to ask a question
@app.post("/ask")
async def ask(request: QuestionRequest):
    try:
        # Train Vanna AI on the database schema (if not already trained)
        await train_vanna(pool)

        # Ask the question
        result = await ask_question(pool, request.question)

        # Return the result
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)