from fastapi import APIRouter, HTTPException
from acp_sdk import AgentManifest
from acp_config import ACPConfig

# Create router for agent endpoints
router = APIRouter(prefix="/agents", tags=["agents"])

@router.get("")
def list_agents():
    """List all registered agents with full metadata"""
    agents = [
        AgentManifest(
            name="ecommerce-magentic-one",
            description="Intelligent e-commerce agent powered by Microsoft's Magentic-One multi-agent orchestrator",
            input_content_types=["text/plain", "application/json"],
            output_content_types=["text/plain", "application/json"],
            metadata={
                "capabilities": [
                    "automatic_query_routing", 
                    "multi_agent_orchestration",
                    "rag_retrieval", 
                    "sql_querying", 
                    "response_composition"
                ],
                "orchestrator": "Microsoft Magentic-One",
                "natural_languages": ["en"],
                "max_context_length": ACPConfig.AGENT_MAX_CONTEXT_LENGTH,
                "supports_streaming": True,
                "agents": [
                    {"name": "EcommerceInfoAgent", "role": "RAG and document retrieval"},
                    {"name": "EcommerceDataAgent", "role": "SQL database queries"}
                ]
            }
        )
    ]
    
    return {
        "agents": [agent.model_dump() for agent in agents],
        "count": len(agents),
        "server_info": {
            "id": ACPConfig.SERVER_ID,
            "name": ACPConfig.SERVER_NAME,
            "version": ACPConfig.SERVER_VERSION
        }
    }

@router.get("/{agent_name}")
def get_agent(agent_name: str):
    """Get specific agent information"""
    if agent_name == "ecommerce-magentic-one" or agent_name == "ecommerce-router" or agent_name == "router":
        return AgentManifest(
            name="ecommerce-magentic-one",
            description="Intelligent e-commerce agent powered by Microsoft's Magentic-One multi-agent orchestrator",
            input_content_types=["text/plain", "application/json"],
            output_content_types=["text/plain", "application/json"],
            metadata={
                "capabilities": [
                    "automatic_query_routing", 
                    "multi_agent_orchestration",
                    "rag_retrieval", 
                    "sql_querying", 
                    "response_composition"
                ],
                "orchestrator": "Microsoft Magentic-One",
                "natural_languages": ["en"],
                "max_context_length": ACPConfig.AGENT_MAX_CONTEXT_LENGTH,
                "supports_streaming": True,
                "agents": [
                    {"name": "EcommerceInfoAgent", "role": "RAG and document retrieval"},
                    {"name": "EcommerceDataAgent", "role": "SQL database queries"}
                ]
            }
        ).model_dump()
    else:
        raise HTTPException(404, f"Agent '{agent_name}' not found")
