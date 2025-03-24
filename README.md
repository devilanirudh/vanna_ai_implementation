# Vanna AI Implementation

This is a FastAPI application that uses Vanna AI to answer natural language questions about your database.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with the following variables:
   ```
   VANNA_API_KEY=your_vanna_api_key
   VANNA_MODEL_NAME=prodloop  # Optional, defaults to 'prodloop'
   ```

## Running the Application

1. Start the FastAPI server:
   ```bash
   python main.py
   ```
2. The server will start on `http://localhost:8000`

## API Usage

Send a POST request to `/ask` endpoint with a JSON body containing your question:

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the top 5 reviews?"}'
```

## Environment Variables

- `VANNA_API_KEY`: Your Vanna AI API key (required)
- `VANNA_MODEL_NAME`: The Vanna AI model to use (optional, defaults to 'prodloop')

## Security Note

Make sure to never commit your `.env` file or expose your API keys. The `.env` file is included in `.gitignore` by default. 