import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo 
load_dotenv()
api = os.getenv('GROQ_API_KEY')
model = Groq(id='DeepSeek-R1-Distill-Llama-70B', api_key=api)
web_search_agent = Agent(
    name = 'Web search Agent',
    role = 'Search the web for the information',
    model = model,
    tools = [DuckDuckGo()],
    instructions = ['Always provide the sources'],
    show_tools_calls = True,
    markdown = True
)

finance_agent = Agent(
    name = 'Finance AI Agent',
    model = model,
    tools = [YFinanceTools(
        stock_price = True, 
        analyst_recommendations = True, 
        stock_fundamentals = True
    )],
    instructions = ['Use tables to display the data'],
    show_tools_calls = True,
    markdown = True,
)


multimodal_ai_agent = Agent(
    team = [web_search_agent,finance_agent],
    instructions = ['Always provide the sources','Use tables to display the data'],
    model = model,
    show_tools_calls = True,
    markdown = True
)

multimodal_ai_agent.print_response(
    "summarise analyst recommendations and share the latest news for NVDA",
    stream=True
)



