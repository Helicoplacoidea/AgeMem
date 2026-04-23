#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgeMem (AgentScope): CLI to run the memory agent.

"""
import asyncio
import os
from typing import Tuple

from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import ChatModelBase

try:
    from agentscope.model import DashScopeChatModel
except Exception:
    OpenAIChatModel = DashScopeChatModel = None

from .agent import AgeMem

async def main() -> None:
    model = DashScopeChatModel(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model_name=os.getenv("AGENT_MODEL_NAME") or "qwen-max",
        enable_thinking=False,
        stream=True,
    )

    formatter = DashScopeChatFormatter()
    sys_prompt = (
        "You are an intelligent assistant that solves complex problems by managing context and long-term memory with tools. "
        "Your job is to capture and organize any information that is helpful, relevant, or useful for the user or for solving their problems—facts, preferences, intermediate results, key decisions, and follow-up needs. "
        "Use tools to: summarize context when it gets long, clear context when starting fresh, retrieve relevant memories, add new useful information to memory, update existing memories when things change, and delete memories when they are no longer needed. "
        "Be concise and helpful; proactively store and recall what matters for the user."
    )

    agent = AgeMem(
        name="AgeMem",
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
    )

    print("=== AgeMem (AgentScope) ===\n")
    print("Type your message and press Enter. 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("[user]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye.")
            break
        reply_msg = await agent.reply(msg=Msg(name="user", content=user_input, role="user"))
        print(f"[agent] {reply_msg.get_text_content()}\n")


if __name__ == "__main__":
    asyncio.run(main())
