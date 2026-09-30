CLASSIFIER_PROMPT = """
Classify the user's latest message into exactly one intent:

- recommend: The user wants recommendations or suggestions for movies or TV
  shows, including requests based on genre, mood, era, runtime, or preferences.
- title_question: The user asks about, discusses, or gives an opinion about a
  specific movie or TV show.
- general: Greetings, ordinary conversation, and questions unrelated to
  recommendations or a specific movie or TV show.

Extract recommendation filters only when they are relevant.

Supported filters:
- media_type: "movie", "tv", or "both"
- genres: a list of genre names
- year: one specific release year
- year_from: earliest release year
- year_to: latest release year
- max_runtime: maximum runtime in minutes, for movies
- query: a specific title or search phrase, only when the user wants results
  matching that title or phrase

Do not invent filters that the user did not express.
Use null for filters when none apply.
Classify based primarily on the user's latest message. Use earlier messages
only when needed to understand references such as "more like those".
"""
