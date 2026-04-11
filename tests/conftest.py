import pytest
from anime_vid_generator.config import Stage1Config
from anime_vid_generator.workflow.builder import WorkflowBuilder


@pytest.fixture
def stage1_config() -> Stage1Config:
    return Stage1Config()


@pytest.fixture
def builder() -> WorkflowBuilder:
    return WorkflowBuilder()
