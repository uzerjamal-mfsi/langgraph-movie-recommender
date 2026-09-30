# LangGraph Movie & TV Recommender

A conversational movie and TV recommender built with LangGraph, OpenRouter, TMDB, and SQLite. The application is chat-only: you describe what you want to watch, ask about titles, or share your opinions in natural language. There are no commands for managing the library or preferences.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- An OpenRouter API key
- A TMDB API key

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Create your local environment file from the example:

   ```bash
   cp .env.example .env
   ```

   On Windows PowerShell, use `Copy-Item .env.example .env`.

3. Fill in the required values in `.env`

## Run the chat

```bash
uv run movie-recommender
```

Resume a previous conversation thread by passing its ID:

```bash
uv run movie-recommender --thread-id <thread-id>
```

The CLI prints a session thread ID at startup. Type `exit` or `quit` to leave, or press `Ctrl+C`.

## Docker

The Docker setup mounts `./data` into the container so SQLite files and other local data survive restarts.

1. Create `.env` as described above.
2. Build and start the interactive chat:

   ```bash
   docker compose up --build
   ```

You can also run a one-off session:

```bash
docker compose run --rm app
```

## Tests

Run the smoke tests without external API calls:

```bash
uv run pytest
```