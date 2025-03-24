import os
from typing import List, Optional

import typer
from dotenv import load_dotenv
from phi.assistant import Assistant
from phi.embedder.google import GeminiEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.llm.groq import Groq 
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.vectordb.pgvector import PgVector2

load_dotenv()
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(
        collection="recipes",
        db_url=db_url,
        embedder=GeminiEmbedder(api_key=os.getenv("GOOGLE_API_KEY")),
    ),
)

knowledge_base.load()
storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)



def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None
    if not new:
        existing_run_ids = storage.get_all_run_ids() 

        if existing_run_ids:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id,
        user=user,
        llm=Groq(id="DeepSeek-R1-Distill-Llama-70B", api_key=os.getenv("GROQ_API_KEY")),
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)


if __name__ == "__main__":
    typer.run(pdf_assistant)
