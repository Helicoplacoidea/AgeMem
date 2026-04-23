# AgeMem (AgentScope)

Standalone release of the **AgeMem** agent: a ReAct-style agent with **6 tools** for self-managing **short-term context** and **long-term memory**, built on [AgentScope](https://github.com/modelscope/agentscope).

## Features

- **6 tools**: `summary_context`, `clear_context`(`filter_context`), `retrieve_memory`, `add_memory`, `update_memory`, `delete_memory`
- **Long-term memory**: in-memory vector store with embedding-based retrieval (DashScope text-embedding by default)

### The 6 Memory Tools

| Tool | Type | Description |
|------|------|-------------|
| `summary_context` | Short-term | Compress selected conversation rounds into a summary |
| `clear_context` / `filter_context` | Short-term | Remove irrelevant messages by similarity |
| `retrieve_memory` | Short-term | Pull relevant entries from long-term memory into context |
| `add_memory` | Long-term | Store new information in the external vector store |
| `update_memory` | Long-term | Update an existing memory entry |
| `delete_memory` | Long-term | Delete an obsolete memory entry |

## Install

From the folder containing `AgeMem_code_agentscope` (e.g. project root):

```bash
pip install -r AgeMem_code_agentscope/requirements.txt
```

## Run

From the **parent directory** of `AgeMem_code_agentscope` (so that `AgeMem_code_agentscope` is a package):

```bash
python -m AgeMem_code_agentscope.main
```

Example (DashScope):

```bash
export DASHSCOPE_API_KEY=your_key
python -m AgeMem_code_agentscope.main
```

## Configuration (environment variables)

| Variable | Description |
|----------|-------------|
| `AGENT_MODEL_NAME` | Model name (e.g. `qwen-max`) |
| `DASHSCOPE_API_KEY` | Api key |

## Layout

```
AgeMem_code_agentscope/
  __init__.py    # Package exports (AgeMem, memory, prompts)
  main.py        # Entry point (CLI), model building (DashScope / OpenAI only)
  agent.py       # AgeMem (ReAct agent + 6 tools)
  memory.py      # AgentScopeLongtermMemory, InMemoryVectorStore
  prompts.py     # SUMMARY_CONTEXT_SYS_PROMPT, TEXT_SIMILARITY_SYS_PROMPT
  src/           # Helpers: utils, llm_client, schemas, hooks
  requirements.txt
  README.md
```
