"""OpenAI-style tool schemas for the BrowseComp-Plus rollout.

These are consumed by `tokenizer.apply_chat_template(messages, tools=TOOLS, ...)`
so that Qwen3.5's chat template renders the canonical `<tools>...</tools>` block
plus the `<tool_call><function=...></function></tool_call>` format instructions
into the system prompt automatically.

Descriptions are lifted verbatim from the BrowseComp-Plus reference parquet
system prompt (SUPO paper, arXiv 2510.11967) so behavior stays identical to the
reference implementation apart from the tool-call wrapper format.
"""

from __future__ import annotations

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Performs a web search: supply a string 'query' and optional 'topk'. "
            "The tool retrieves the top 'topk' results (default 10) for the query, "
            "returning their docid, url, and document content "
            "(may be truncated based on token limits)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query string for the search.",
                },
                "topk": {
                    "type": "integer",
                    "description": "Return the top k pages.",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
}

OPEN_PAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "open_page",
        "description": (
            "Open a page by docid or URL and return the complete content. "
            "Provide either 'docid' or 'url'; if both are provided, prefer 'docid'. "
            "The docid or URL must come from prior search tool results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "docid": {
                    "type": "string",
                    "description": "Document ID from search results to resolve and fetch.",
                },
                "url": {
                    "type": "string",
                    "description": "Absolute URL from search results to fetch.",
                },
            },
            "required": [],
        },
    },
}

FINISH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Return the final result when you have a definitive answer or cannot "
            "progress further. Provide a concise answer plus a brief, "
            "evidence-grounded explanation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A succinct, final answer.",
                },
                "explanation": {
                    "type": "string",
                    "description": (
                        "A brief explanation for your final answer. For this section "
                        "only, cite evidence documents inline by placing their docids "
                        "in square brackets at the end of sentences (e.g., [20]). "
                        "Do not include citations anywhere else."
                    ),
                },
                "confidence": {
                    "type": "string",
                    "description": "Your confidence score between 0% and 100% for your answer.",
                },
            },
            "required": ["answer", "explanation"],
        },
    },
}

TOOLS = [SEARCH_SCHEMA, OPEN_PAGE_SCHEMA, FINISH_SCHEMA]

# Future-facing: compress is not yet exposed to the model. When we later
# support model-driven compression (policy decides *when* to summarize), we
# will add this to TOOLS and _run_action will grow a `compress` branch that
# routes the model's summary through the same _start_new_subtrajectory hook
# the heuristic path already uses. For now the heuristic in generate() emits
# a bare <summary>...</summary> block instead of a <tool_call>, but the sub-
# trajectory data model is identical, so migration will be a small delta.
COMPRESS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compress",
        "description": (
            "Compress the context by producing a summary of the work so far. Use "
            "this when the context is getting long and you want to continue with "
            "just the essential state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "A comprehensive summary of the work so far, including the "
                        "original question, key verified findings with docids, and "
                        "the next tactical step."
                    ),
                },
            },
            "required": ["summary"],
        },
    },
}

QWEN_SYSTEM_PROMPT = (
    "You are an expert research agent focused on comprehensive research strategy, "
    "execution, and final report writing. Your core goal is to be maximally helpful "
    "to the user by researching their query thoroughly and creating an excellent "
    "research report that answers the query very well."
)
