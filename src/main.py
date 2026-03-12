"""
学习助手 Agent - 主入口
"""
import os
import sys
import uuid
import warnings
from typing import Dict, Any

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph.graph import create_graph
from src.graph.state import AgentState
from src.config import KB_SUMMARY_FILE
from src.processing.kb_summary import KnowledgeBaseSummary


class LearningAssistant:
    """学习助手 Agent"""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.kb_summary = KnowledgeBaseSummary()
        self.app = create_graph().compile(checkpointer=self.checkpointer)
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    def run(self):
        """运行学习助手"""
        print("=" * 60)
        print("📚 学习助手 Agent")
        print("=" * 60)
        print("\n我可以帮助你学习任何主题！")
        print("- 支持多轮对话")
        print("- 自动搜索和下载学习材料")
        print("- 基于知识库生成带有引用的回答")
        print("\n输入 'exit'、'quit' 或 '退出' 结束对话")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("\n💬 请输入你的问题: ").strip()

                if not user_input:
                    continue

                initial_state: AgentState = {
                    "messages": [{"role": "user", "content": user_input}],
                    "user_query": user_input,
                }

                final_state = self._execute_graph(initial_state, self.config)

                # 检查是否要退出
                if final_state and final_state.get("should_exit", False):
                    break

            except KeyboardInterrupt:
                print("\n\n👋 感谢使用学习助手，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()

    def _execute_graph(self, initial_state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
        """执行状态图，返回最终状态"""
        final_state = None

        for event in self.app.stream(initial_state, config, stream_mode="updates"):
            # 更新状态
            for node_name, node_output in event.items():
                if node_name != "__interrupt__":
                    if final_state is None:
                        final_state = initial_state.copy()
                    final_state.update(node_output)

            self._handle_event(event)

            if "__interrupt__" in event:
                self._handle_interrupt(event, config)

        return final_state or initial_state

    def _handle_event(self, event: Dict):
        """处理状态图事件"""
        for node_name, node_output in event.items():
            if node_name == "generate_answer":
                answer = node_output.get("answer", "")
                if answer:
                    print("\n" + "=" * 60, flush=True)
                    print("🤖 助手回答:", flush=True)
                    print("=" * 60, flush=True)
                    print(answer, flush=True)
                    print("=" * 60, flush=True)

    def _handle_interrupt(self, event: Dict, config: Dict[str, Any]):
        """处理 interrupt 事件"""
        sys.stdout.flush()

        interrupt_data = event.get("__interrupt__", {})
        if isinstance(interrupt_data, tuple) and len(interrupt_data) > 0:
            interrupt_info = interrupt_data[0]
        else:
            interrupt_info = interrupt_data

        prompt = "你的选择: "
        if isinstance(interrupt_info, dict):
            prompt = interrupt_info.get("prompt", prompt)

        user_input = input(prompt).strip()

        for event in self.app.stream(Command(resume=user_input), config, stream_mode="updates"):
            self._handle_event(event)

            if "__interrupt__" in event:
                self._handle_interrupt(event, config)


def main():
    assistant = LearningAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
