import logging
import textwrap

import numpy as np

from eaa_core.tool.coding import BashCodingTool, SimplePythonEvalTool, PythonCodingTool

import test_utils as tutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestCodingTool(tutils.BaseTester):
    def test_literal_eval_tool_evaluates_literal(self):
        tool = SimplePythonEvalTool()

        result = tool.evaluate("{'values': [1, 2, 3], 'enabled': True}")

        assert result == {"values": [1, 2, 3], "enabled": True}

    def test_literal_eval_tool_evaluates_arithmetic(self):
        tool = SimplePythonEvalTool()

        result = tool.evaluate(
            "(273.77 + ((290.77 - 291.835) - 3.5), "
            "235.58 + ((270.58 - 271.52) - 3.5))"
        )

        np.testing.assert_allclose(result, (269.205, 231.14))

    def test_literal_eval_tool_evaluates_sum(self):
        tool = SimplePythonEvalTool()

        result = tool.evaluate("sum([1, 2 * 3, -4])")

        assert result == 3

    def test_literal_eval_tool_rejects_other_calls(self):
        tool = SimplePythonEvalTool()

        with np.testing.assert_raises(ValueError):
            tool.evaluate("__import__('os').system('echo unsafe')")

    def test_literal_eval_tool_rejects_oversized_power(self):
        tool = SimplePythonEvalTool()

        with np.testing.assert_raises(ValueError):
            tool.evaluate("2 ** 10000")

    def test_literal_eval_tool_does_not_require_approval(self):
        tool = SimplePythonEvalTool()

        assert tool.require_approval is False
        assert tool.exposed_tools[0].require_approval is False

    def test_execute_code_calculates_mean(self):
        tool = PythonCodingTool()
        code = textwrap.dedent("""\
            import numpy as np
            print(np.mean(np.arange(10)))
        """)

        result = tool.execute_code(code)
        if self.debug:
            print(result)

        assert result["returncode"] == 0
        assert result["timeout"] is False
        assert result["stderr"] == ""
        assert result["stdout"].strip() == str(np.mean(np.arange(10)))

    def test_python_coding_tool_requires_approval_flag(self):
        tool = PythonCodingTool()
        assert tool.require_approval is True

    def test_coding_tool_names_use_execute_action(self):
        assert PythonCodingTool().exposed_tools[0].name == "python_coding_tool.execute"
        assert BashCodingTool().exposed_tools[0].name == "bash_coding_tool.execute"


if __name__ == "__main__":
    tester = TestCodingTool()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_execute_code_calculates_mean()
