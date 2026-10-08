import json
import requests
from pydantic import BaseModel
from typing import List

class Source(BaseModel):
    page: int
    text: str

class AIResponse(BaseModel):
    answer: str
    confidence: int
    sources: List[Source]

class AIAnalyzer:
    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.url = f"{host}/api/generate"

    def ask(self, question: str, sources: List[dict]) -> AIResponse:
        if not sources:
            return AIResponse(
                answer="No relevant document context found to answer the question.",
                confidence=0,
                sources=[]
            )

        context = "\n\n".join(
            [f"Page {s['page']}:\n{s['text']}" for s in sources]
        )

        prompt = f"""You are a legal document analysis assistant.
Answer the user's question ONLY using the provided document context.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

Return valid JSON in this exact structure:
{{
  "answer": "<detailed answer based on the context>",
  "confidence": <integer from 0 to 100>
}}

Do not invent facts. Mention when the available evidence is insufficient.
"""

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            raw_content = response.json().get("response", "{}")
            parsed = json.loads(raw_content)

            answer = parsed.get("answer", "No answer could be generated.")
            confidence = int(parsed.get("confidence", 80))
        except Exception as e:
            answer = f"Error processing response: {str(e)}"
            confidence = 0

        return AIResponse(
            answer=answer,
            confidence=confidence,
            sources=[Source(**s) for s in sources]
        )