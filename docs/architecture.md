# Architecture

It's a chat loop. The CLI reads a line, sends it to a LangGraph state graph, and prints the last assistant message that comes back. The graph classifies what the user is trying to do, runs one branch, and stops. There are no slash commands and no background work.

Each conversation is a thread. The compiled graph is given a SQLite checkpointer, so the state is saved after every step. That is how `--thread-id` can resume an old conversation.

## What each part does

### classify_intent

One structured LLM call. It returns an intent plus optional recommendation filters (media type, genres, year range, runtime, a free text query, and `allow_rewatch`). The prompts are in `graph/prompts.py`.

The five intents are `recommend`, `title_question`, `general`, `record_feedback` and `update_preferences`.

### recommend

Three nodes:

1. `load_taste` reads preferences, watched titles and ratings from SQLite. It builds a small text summary and fills `taste_likes`, `taste_dislikes` and `watched_ids`.
2. `retrieve_taste_memories` does a Qdrant similarity search with the current user message (top 5) and merges anything relevant into the like/dislike lists. If Qdrant is unreachable it just returns nothing, the rest of the graph keeps going.
3. `recommend` asks TMDB for candidates (search when there is a query, discover otherwise), drops titles already watched unless rewatch is allowed, re ranks by word overlap with the taste lists, and formats the top 5.

### answer_general

Both `title_question` and `general` end up here. It's a plain LLM call over the message history.

### record_feedback

- `extract_feedback` pulls out title, media type, year, watched flag, rating (1 to 10), review, and positive/negative aspects as structured output.
- `resolve_title` searches TMDB and filters by exact-ish title and year. Zero matches is an error. One match is saved. More than one interrupts the graph with a numbered list.
- `ask_clarifying_question` calls LangGraph's `interrupt()`, so the graph pauses. The CLI notices `__interrupt__` in the result, prints the question, and sends the next line back as `Command(resume=...)`.
- `resolve_clarification` accepts a number, a title, or `movie` / `tv`. If it still can't decide, it asks again.
- `save_feedback` upserts the library row and creates one inferred preference per aspect. Missing fields keep their old values, so you can rate something later without losing the review.

### update_preferences

`extract_preferences` turns statements like "I usually hate musicals" into explicit preference records, then `save_preferences` upserts them on `(text, kind)` and indexes each one in Qdrant. This branch does not touch the library.

Preferences from feedback are marked `inferred` and keep the source title. The explicit ones don't have a source title. The taste summary doesn't really care, it merges both lists, but the source fields are there if you need them.

## State

`GraphState` is a `TypedDict` in `graph/state.py`. Messages use `add_messages`, everything else is overwritten by whichever node returns it.

The fields are roughly:

- `messages`, the conversation
- `intent` and `filters`, from classification
- `taste_context`, `taste_likes`, `taste_dislikes`, `watched_ids`
- `relevant_taste_memories`, from Qdrant
- `feedback`, `resolved_title`, `saved_summary`
- `clarification_question`, `clarification_candidates`, `clarification_answer`, `needs_clarification`
- `extracted_preferences`
- `error`

Error handling uses the `error` field. Nodes generally catch their own exceptions, store a friendly message there, and routing sends the user to `END` (or to `save_feedback`, which then prints the error). The CLI never sees a raw traceback, it just prints the message.
