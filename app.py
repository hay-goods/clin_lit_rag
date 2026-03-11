"""
app.py — Gradio chat interface for the Clinical Literature RAG Agent
"""

import os
from dotenv import load_dotenv
import anthropic
import gradio as gr

from retriever import Retriever

load_dotenv()

CLAUDE_MODEL = "claude-sonnet-4-6"
TOP_K = 5

retriever = Retriever(top_k=TOP_K)
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = """You are a clinical literature analyst. You help researchers, clinicians, and analysts
find epidemiological estimates and clinical evidence from peer-reviewed publications
and practice guidelines.

You will be given excerpts retrieved from the corpus. Synthesize a precise, evidence-based
answer using ONLY the provided context. For every claim you make, cite the source document
and page number using the format: [Source: <filename>, p.<page>].

If the context does not contain enough information to answer the question, say so clearly
rather than speculating. Do not invent statistics or citations."""


def build_context_block(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"--- Excerpt {i} | Source: {c['source']} | Page: {c['page']} ---\n{c['text']}"
        )
    return "\n\n".join(lines)


def format_citations(chunks: list[dict]) -> str:
    seen = set()
    lines = ["**Sources retrieved:**"]
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            lines.append(f"- {c['source']} — p.{c['page']}")
    return "\n".join(lines)


def chat(message: str, history: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
    if not message.strip():
        return "", history

    # Retrieve relevant chunks
    try:
        chunks = retriever.query(message)
    except Exception as e:
        error_msg = (
            f"⚠️ Retrieval error: {e}\n\n"
            "Make sure you have run `python ingest.py` to populate the corpus."
        )
        history.append((message, error_msg))
        return "", history

    context_block = build_context_block(chunks)
    citations = format_citations(chunks)

    # Build messages for Claude (include chat history for multi-turn)
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({
        "role": "user",
        "content": (
            f"Context excerpts from the clinical literature corpus:\n\n"
            f"{context_block}\n\n"
            f"Question: {message}"
        ),
    })

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    answer = response.content[0].text
    full_response = f"{answer}\n\n---\n{citations}"

    history.append((message, full_response))
    return "", history


def build_ui():
    corpus_size = retriever.corpus_size
    status = (
        f"Corpus loaded — {corpus_size:,} chunks indexed."
        if corpus_size > 0
        else "⚠️ No corpus found. Run `python ingest.py` first."
    )

    with gr.Blocks(title="Clinical Literature RAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## Clinical Literature RAG Agent\n"
            "Ask plain-English questions about your indexed clinical corpus — "
            "epidemiology, trial outcomes, and guideline recommendations. "
            "Answers are synthesized with citations from the source documents."
        )
        gr.Markdown(f"*{status}*")

        chatbot = gr.Chatbot(
            label="Clinical Literature Assistant",
            height=520,
            bubble_full_width=False,
        )
        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="e.g. What is the prevalence of severe aortic stenosis in adults over 65?",
                label="Your question",
                scale=5,
                show_label=False,
            )
            send_btn = gr.Button("Ask", variant="primary", scale=1)

        gr.Examples(
            examples=[
                "What is the prevalence of severe aortic stenosis in adults over 65?",
                "What does the PARTNER 3 trial say about 2-year mortality for TAVR vs surgery?",
                "What fraction of patients with severe AS are currently asymptomatic?",
                "What are the ACC/AHA guideline recommendations for TAVR indication?",
                "How does EVOLUT Low Risk compare to PARTNER 3 on stroke outcomes?",
            ],
            inputs=msg_box,
        )

        clear_btn = gr.Button("Clear conversation", variant="secondary")

        send_btn.click(chat, [msg_box, chatbot], [msg_box, chatbot])
        msg_box.submit(chat, [msg_box, chatbot], [msg_box, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False)
