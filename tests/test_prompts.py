import unittest

from smart_clean_agent.utils.prompt_loader import load_report_prompts, load_system_prompts


class PromptRulesTestCase(unittest.TestCase):
    def test_main_prompt_does_not_require_visible_chain_of_thought(self):
        prompt = load_system_prompts()

        self.assertNotIn("每次调用工具前，必须输出**真实的自然语言思考过程**", prompt)
        self.assertIn("不要向用户展示内部思考过程", prompt)
        self.assertIn("不要使用“结论：”“依据：”“建议：”", prompt)

    def test_report_prompt_does_not_require_visible_chain_of_thought(self):
        prompt = load_report_prompts()

        self.assertIn("不向用户展示内部推理、工具调用步骤", prompt)


if __name__ == "__main__":
    unittest.main()

