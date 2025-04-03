class _Tags:
    """
    Container for predefined tag sets used in pipeline steps.
    
    This class defines standard tag sets that are used to identify
    different types of pipeline steps. Tags are used for filtering,
    searching, and identifying steps in a pipeline.
    """
    STEP_FUNCTION: str = {'function'}
    """Set of tags for function steps."""
    
    STEP_DELETE: str = {'delete'}
    """Set of tags for delete steps."""
    
    STEP_VARIABLES_DICT: str = {'add_variables'}
    """Set of tags for variable dictionary steps."""
    
    STEP_IMPORT_LIB: str = {'import_lib'}
    """Set of tags for import library steps."""
    # Add any other tags as needed


tags = _Tags()
