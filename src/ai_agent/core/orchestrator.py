from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from src.ai_agent.components.prompts.system_prompt import get_system_prompt
from src.config.settings import Settings

settings = Settings()

class AIAgentOrchestrator:
    """Orchestrates a ReAct (Reasoning + Acting) AI agent for multi-tool reasoning.
    
    Manages agent initialization with LangGraph, tool integration, and session persistence
    via MongoDB. Handles message flow, agent execution, and result parsing.
    """
    def __init__(self, model_name: str, tools: list, mongo_path: str, template_prompt_path: str):
        """Initialize the ReAct agent orchestrator with LLM, tools, and session storage.
        
        Sets up the ChatOllama language model, integrates provided tools, loads system prompt,
        and configures MongoDB for conversation history persistence.
        
        Args:
            model_name: Name of the Ollama language model (e.g., 'llama3.1:8b')
            tools: List of LangChain Tool objects for the agent to use
            mongo_path: MongoDB connection string for session state persistence
            template_prompt_path: Path to the system prompt template file
        """
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.tools = tools
        self.mongo_uri = mongo_path

        self.tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])

        self.system_prompt = get_system_prompt(
            tools_desc=self.tools_description,
            tool_names=[t.name for t in self.tools],
            template_path=template_prompt_path
        )

        print("✅ AIAgentOrchestrator initialized successfully\n")

    def invoke(self, session_id: str, user_message: str) -> dict:
        """Execute the ReAct agent loop with the user message and return cleaned response.
        
        Orchestrates the complete agent reasoning cycle:
        1. Creates a ReAct agent with LLM and tools
        2. Sets up MongoDB-backed session state (thread_id)
        3. Constructs message chain with system prompt and user query
        4. Runs agent.invoke() which handles reasoning and tool calls
        5. Parses agent output to extract clean final answer
        6. Filters out internal reasoning and tool observations
        
        Args:
            session_id: Unique session identifier for conversation history persistence (thread_id)
            user_message: The user's query/input to process
        
        Returns:
            Dictionary with key:
            - 'final_answer': Clean final response from the agent (tool observations removed)
        
        Processing flow:
        - Agent thinks (internal reasoning)
        - Agent selects and calls tools as needed
        - Agent observes tool results
        - Agent formulates final answer
        - Parser extracts only the final answer text
        
        Raises:
            Returns error message in result if message parsing fails
        """

        with MongoDBSaver.from_conn_string(self.mongo_uri) as checkpointer:

            agent = create_react_agent(
                model=self.llm,
                tools=self.tools,
                checkpointer=checkpointer
            )

            config = {"configurable": {"thread_id": session_id}}

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_message)
            ]

            result = agent.invoke(
                {"messages": messages},
                config=config,
                debug=False
            )

            if "messages" not in result or not result["messages"]:
                return {
                    "final_answer": "Sorry, a system error occurred. Please try again."
                }

            conversation = result["messages"]
            last_human_index = None
            for i in range(len(conversation) - 1, -1, -1):
                if isinstance(conversation[i], HumanMessage):
                    last_human_index = i
                    break

            thought_lines = []

            start_index = (last_human_index + 1) if last_human_index else 2

            for msg in conversation[start_index:-1]:
                if isinstance(msg, AIMessage):
                    content = msg.content
                    if content:
                        thought_lines.append(f"Agent (Thought): {content}")
                elif isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, 'name', 'Unknown')
                    thought_lines.append(f"Observation ({tool_name}): {msg.content}")

            full_thought_process = "\n\n".join(thought_lines)

            if not full_thought_process:
                full_thought_process = "Agent is processing the request..."

            last_message = conversation[-1]
            raw_content = last_message.content

            if "Final Answer:" in raw_content:
                clean_final_answer = raw_content.split("Final Answer:", 1)[1].strip()
            else:
                clean_final_answer = raw_content.strip()

            clean_final_answer = clean_final_answer.split("Thought:")[0].strip()

            return {
                "final_answer": clean_final_answer
            }