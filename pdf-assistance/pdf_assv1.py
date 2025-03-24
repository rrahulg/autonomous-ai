import os

from phi.agent import Agent
from phi.embedder.google import GeminiEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.model.groq import Groq
from phi.vectordb.pgvector import PgVector, SearchType

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=GeminiEmbedder(api_key=os.getenv("GOOGLE_API_KEY")),
    )
)

knowledge_base.load(recreate=True, upsert=True)

agent = Agent(
    model=Groq(id="DeepSeek-R1-Distill-Llama-70B", api_key=os.getenv("GROQ_API_KEY")),
    knowledge=knowledge_base,
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True,
)

agent.print_response(
    "How do I make chicken and galangal in coconut milk soup", stream=True
)
agent.print_response("What was my last question?", stream=True)
