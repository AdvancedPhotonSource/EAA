
import argparse
import logging
import os

import pytest
import numpy as np

from eaa.agents.base import print_message, generate_openai_message
from eaa.agents.openai import OpenAIAgent
from eaa.tools.base import ToolReturnType
from eaa.task_managers.base import BaseTaskManager

import test_utils as tutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestPurgeImageMessages(tutils.BaseTester):
    
    def test_purge_image_messages_keep_text(self):
        context = [
            generate_openai_message(
                content="This is message 1",
                role="user",
                image=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            ),
            {
                "role": "assistant",
                "content": "This is message 2",
            },
            generate_openai_message(
                content="This is message 3",
                role="user",
                image=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            )
        ]
        task = BaseTaskManager()
        task.context = context
        task.purge_context_images(keep_first_n=1, keep_last_n=0)
        
        answer = [
            context[0],
            context[1],
            {
                "role": "user",
                "content": "This is message 3\n<image> \n",
            }
        ]
        assert task.context == answer
        return
    
    def test_purge_image_messages_remove_completely(self):
        context = [
            generate_openai_message(
                content="This is message 1",
                role="user",
                image=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            ),
            {
                "role": "assistant",
                "content": "This is message 2",
            },
            generate_openai_message(
                content="This is message 3",
                role="user",
                image=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            )
        ]
        task = BaseTaskManager()
        task.context = context
        task.purge_context_images(keep_first_n=1, keep_last_n=0, keep_text=False)
        
        answer = [
            context[0],
            context[1],
        ]
        assert task.context == answer
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    tester = TestPurgeImageMessages()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_purge_image_messages_keep_text()
    tester.test_purge_image_messages_remove_completely()
