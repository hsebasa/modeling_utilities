
from typing import Set


class _Tags:
    """
    Container for predefined tag sets used in pipeline steps.
    
    This class defines standard tag sets that are used to identify
    different types of pipeline steps. Tags are used for filtering,
    searching, and identifying steps in a pipeline.
    """
    STEP_FUNCTION: Set[str] = {'function'}
    """Set of tags for function steps."""
    
    STEP_DELETE: Set[str] = {'delete'}
    """Set of tags for delete steps."""
    
    STEP_VARIABLES_DICT: Set[str] = {'add_variables'}
    """Set of tags for variable dictionary steps."""
    
    STEP_IMPORT_LIB: Set[str] = {'import_lib'}
    """Set of tags for import library steps."""
    
    STEP_MODEL_TRAIN_ONLY: Set[str] = {'model_train_only'}
    STEP_MODEL_INSTANCE: Set[str] = STEP_MODEL_TRAIN_ONLY|{'model_instance'}
    STEP_MODEL_FIT: Set[str] = STEP_MODEL_TRAIN_ONLY|{'model_fit'}
    STEP_MODEL_PREDICT: Set[str] = {'model_predict'}
    STEP_MODEL_SCORE: Set[str] = {'model_score'}
    """Set of tags for models."""

    # Add any other tags as needed

tags = _Tags()
