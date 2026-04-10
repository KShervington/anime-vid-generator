import pytest
from anime_vid_generator.config import Stage1Config, Stage2Config
from anime_vid_generator.workflow.builder import WorkflowBuilder


@pytest.fixture
def stage1_config() -> Stage1Config:
    return Stage1Config()


@pytest.fixture
def stage2_config() -> Stage2Config:
    return Stage2Config()


@pytest.fixture
def builder() -> WorkflowBuilder:
    return WorkflowBuilder()
