import logging
import textwrap

import numpy as np

from eaa.tools.coding import PythonCodingTool

import test_utils as tutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestCodingTool(tutils.BaseTester):
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


if __name__ == "__main__":
    tester = TestCodingTool()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_execute_code_calculates_mean()
