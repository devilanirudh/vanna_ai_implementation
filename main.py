import vanna as vn
import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from vanna.remote import VannaDefault
from database import get_review_pool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Vanna AI setup
api_key = os.getenv('VANNA_API_KEY')
vanna_model_name = os.getenv('VANNA_MODEL_NAME') # Default to prodloop if not specified

if not api_key:
    raise ValueError("VANNA_API_KEY environment variable is not set")

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