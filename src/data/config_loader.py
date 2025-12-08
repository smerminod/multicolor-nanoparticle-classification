from pathlib import Path
from typing import Any, Optional
import sys

import yaml
from pydantic import ValidationError

from src.data.schemas import (
    ClassDefinitions,
    PipelineConfig,
    ExperimentsDatabase,
    BaseExperiment
)

# Type alias for validated configuration tuple
ValidatedConfigs = tuple[ClassDefinitions, PipelineConfig, ExperimentsDatabase]

# Project root directory (3 levels up: config_loader.py -> data/ -> src/ -> project_root/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_class_definitions() -> dict[str, Any]:
    """Loads class_definitions.yaml containing nanoparticle composition database."""
    class_path = PROJECT_ROOT / 'class_definitions.yaml'
    if not class_path.exists():
        raise FileNotFoundError(f"Class definitions file not found at: {class_path}")
    with open(class_path, 'r') as file:
        return yaml.safe_load(file)

def load_pipeline_config() -> dict[str, Any]:
    """
    Loads config.yaml containing reusable configuration templates (in the future, 
    will also include analysis parameters).
    """
    config_path = PROJECT_ROOT / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline configuration file not found at: {config_path}")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_experiments_db() -> dict[str, Any]:
    """Loads experiments.yaml containing experiment metadata."""
    db_path = PROJECT_ROOT / 'experiments.yaml'
    if not db_path.exists():
        raise FileNotFoundError(f"Experiments database file not found at: {db_path}")
    with open(db_path, 'r') as file:
        return yaml.safe_load(file)
    
def verify_class_definitions() -> ClassDefinitions:
    """
    Validates class_definitions.yaml schema and data integrity using Pydantic.

    Returns:
        ClassDefinitions: Validated class definitions object

    Raises:
        ValidationError: If class definitions are invalid
        FileNotFoundError: If class_definitions.yaml not found
    """
    print("Verifying class definitions...")
    class_defs_dict = load_class_definitions()

    try:
        validated = ClassDefinitions.model_validate(class_defs_dict)
        print("✅ Class definitions (class_definitions.yaml) verified successfully.")
        return validated
    except ValidationError as e:
        raise ValueError(f"Invalid class definitions (class_definitions.yaml): {e}") from e

def verify_pipeline_config() -> PipelineConfig:
    """
    Validates config.yaml schema and structure using Pydantic.

    Validates:
        - Required template names exist (default_raw_config, default_processed_config, etc.)
        - Hardware-constant acquisition types (CL_SE, SEM) are present
        - All channels/detectors have required configuration (source_prefix)
        - Metadata templates have {id} placeholder

    Returns:
        PipelineConfig: Validated pipeline configuration object

    Raises:
        ValidationError: If config.yaml is invalid
        FileNotFoundError: If config.yaml not found
    """
    print("Verifying pipeline configuration...")
    config_dict = load_pipeline_config()

    try:
        validated = PipelineConfig.model_validate(config_dict)
        print("✅ Pipeline configuration (config.yaml) verified successfully.")
        return validated
    except ValidationError as e:
        raise ValueError(f"Invalid pipeline configuration (config.yaml): {e}") from e

def verify_experiment_config() -> ExperimentsDatabase:
    """
    Validates experiments.yaml structure and data constraints using Pydantic.

    Validates:
        - experiments.yaml structure (project_constants, training_data, inference_data)
        - All experiment fields (beam_current, composition_label, mixture_priors, etc.)
        - Dataset-specific constraints (training vs inference)
        - Mixture priors expansion and sum validation

    Does not cross-validate references to other config files.
    For complete validation with cross-checks, use verify_experiment_references().

    Returns:
        ExperimentsDatabase: Validated experiments database object

    Raises:
        ValidationError: If experiments.yaml structure or data is invalid
        FileNotFoundError: If experiments.yaml not found

    Example:
        >>> experiments_db = verify_experiment_config()
        >>> training_ds = experiments_db.get_training_dataset()
    """
    print("Verifying experiments database structure...")

    experiments_dict = load_experiments_db()
    try:
        experiments_db = ExperimentsDatabase.model_validate(experiments_dict)
        print("✅ Experiments database (experiments.yaml) verified successfully.")
        return experiments_db
    except ValidationError as e:
        raise ValueError(f"Invalid experiments database (experiments.yaml): {e}") from e

def verify_experiment_references(verbose: bool = True) -> ValidatedConfigs:
    """
    Comprehensive validation: validates all three YAML files and cross-references.

    This is the recommended function for complete validation. It:
        1. Validates class_definitions.yaml (via verify_class_definitions)
        2. Validates config.yaml (via verify_pipeline_config)
        3. Validates experiments.yaml (via verify_experiment_config)
        4. Cross-validates references:
            - composition_label exists in class_definitions.yaml
            - possible_composition_labels exist in class_definitions.yaml
            - raw_config, processed_config, metadata_templates exist in config.yaml

    Args:
        verbose: If True (default), prints validation progress. If False, runs silently.

    Returns:
        Tuple of (ClassDefinitions, PipelineConfig, ExperimentsDatabase)
        All three configs are individually validated and guaranteed consistent
        with each other.

    Raises:
        ValueError: If cross-references are broken
        ValidationError: If any YAML file is invalid
        FileNotFoundError: If any YAML file not found

    Examples:
        >>> # Verbose mode (default) - prints validation progress
        >>> validated_configs = verify_experiment_references()

        >>> # Silent mode - no output, useful when embedding in other tools
        >>> validated_configs = verify_experiment_references(verbose=False)

        >>> # Unpacking
        >>> class_defs, pipeline_config, experiments_db = verify_experiment_references()
    """
    if verbose:
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION: All config files + cross-references")
        print("="*80 + "\n")

    # Validate individual file structures (these print their own messages)
    # If verbose = False, suppress these outputs by temporarily redirecting
    if not verbose:
        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            class_defs = verify_class_definitions()
            pipeline_config = verify_pipeline_config()
            experiments_db = verify_experiment_config()
    else:
        class_defs = verify_class_definitions()
        pipeline_config = verify_pipeline_config()
        experiments_db = verify_experiment_config()

    # Cross-validate references between files
    if verbose:
        print("\nCross-validating references between config files...")

    available_classes = set(class_defs.nanoparticle_classes.keys())
    available_ref_templates = set(pipeline_config.definitions.keys())

    # Compute config reference fields once (shared by all experiments, e.g., 'raw_config', etc)
    ref_fields = [
        name for name, field in BaseExperiment.model_fields.items()
        if isinstance(field.json_schema_extra, dict) and field.json_schema_extra.get('is_config_ref')
    ]

    # Validate training experiments
    if verbose:
        print("  → Validating training experiment references...")
    training_ds = experiments_db.get_training_dataset()
    for group_name in training_ds.get_group_names():
        for exp_id, exp_config in training_ds.get_group_experiments(group_name).items():
            # Check composition_label exists (in class_definitions.yaml)
            comp_label = exp_config.composition_label
            if comp_label not in available_classes:
                raise ValueError(
                    f"Training experiment '{exp_id}' references composition_label "
                    f"'{comp_label}' which doesn't exist in class_definitions.yaml. "
                    f"Available classes: {sorted(available_classes)}"
                )

            # Check config template references (to config.yaml)
            _validate_refs_to_templates(exp_id, exp_config, available_ref_templates, ref_fields)

    # Validate inference experiments
    if verbose:
        print("  → Validating inference experiment references...")
    inference_ds = experiments_db.get_inference_dataset()
    for group_name in inference_ds.get_group_names():
        for exp_id, exp_config in inference_ds.get_group_experiments(group_name).items():
            # Check all possible_composition_labels exist (in class_definitions.yaml)
            for label in exp_config.possible_composition_labels:
                if label not in available_classes:
                    raise ValueError(
                        f"Inference experiment '{exp_id}' references possible_composition_label "
                        f"'{label}' which doesn't exist in class_definitions.yaml. "
                        f"Available classes: {sorted(available_classes)}"
                    )

            # Check config template references (to config.yaml)
            _validate_refs_to_templates(exp_id, exp_config, available_ref_templates, ref_fields)

    if verbose:
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE VALIDATION COMPLETE")
        print("="*80 + "\n")

    return class_defs, pipeline_config, experiments_db

def _validate_refs_to_templates(
    exp_id: str,
    exp_config: Any,
    available_templates: set[str],
    ref_fields: list[str]
) -> None:
    """
    Helper to validate config template references for one experiment.

    Args:
        exp_id: Experiment identifier
        exp_config: Experiment configuration object
        available_templates: Set of available template names from config.yaml
        ref_fields: List of field names that are config references (computed from BaseExperiment schema)
    """
    for ref_field in ref_fields:
        ref_name = getattr(exp_config, ref_field)
        if ref_name not in available_templates:
            raise ValueError(
                f"Experiment '{exp_id}' references {ref_field} '{ref_name}' "
                f"which doesn't exist in config.yaml definitions. "
                f"Available templates: {sorted(available_templates)}"
            )

def get_all_experiments() -> dict[str, dict[str, list[str]]]:
    """
    Convenience function to get all experiment IDs organized by dataset type and group.

    Returns:
        Nested dictionary with structure:
        {
            'training_data': {'group_name': ['exp_id1', 'exp_id2', ...]},
            'inference_data': {'group_name': ['exp_id1', 'exp_id2', ...]}
        }

    Note:
        This is a convenience wrapper around ExperimentsDatabase.get_all_experiments().
        Internally validates experiments.yaml using Pydantic schemas.

        For better performance with repeated calls, use:
            experiments_db = verify_experiment_references()
            all_exp_ids = experiments_db.get_all_experiments()
    """
    experiments_dict = load_experiments_db()
    experiments_db = ExperimentsDatabase.model_validate(experiments_dict)
    return experiments_db.get_all_experiments()

def format_experiments_summary(
    validated_configs: Optional[ValidatedConfigs] = None,
    include_stats: bool = True
) -> str:
    """
    Formats all available experiments as a human-readable string.

    This function provides a formatted overview of all experiments in the database,
    organized by dataset type and group. Useful for CLI tools, notebooks, and logs.

    Args:
        validated_configs: Optional pre-validated configs from verify_experiment_references().
                          If None, loads and validates experiments.yaml automatically.
        include_stats: Whether to include summary statistics (default: True)

    Returns:
        Formatted multi-line string showing all experiments organized hierarchically

    Example:
        >>> # Simple usage (validates each time)
        >>> print(format_experiments_summary())
        📁 Training Data:
           └─ physical_single_particle:
              • 2024-06-29_Dy100
              • 2024-06-29_Ho100
           └─ simulation_single_particle:
              (empty)

        📁 Inference Data:
           └─ physical_multi_particle:
              • 2025-07-02_6-component-mixture

        📊 Summary: 2 training experiment(s), 1 inference experiment(s)

        >>> # Optimized usage with pre-validated configs and no stats summary
        >>> validated = verify_experiment_references()
        >>> summary = format_experiments_summary(validated, include_stats=False)
    """
    # Get experiments database
    if validated_configs is not None:
        _, _, experiments_db = validated_configs
    else:
        experiments_dict = load_experiments_db()
        experiments_db = ExperimentsDatabase.model_validate(experiments_dict)

    all_experiments = experiments_db.get_all_experiments()
    lines = []

    # Format experiments by dataset type
    for dataset_type in ['training_data', 'inference_data']:
        if all_experiments[dataset_type]:
            lines.append(f"\n📁 {dataset_type.replace('_', ' ').title()}:")
            for group_name, exp_ids in all_experiments[dataset_type].items():
                if exp_ids:
                    lines.append(f"   └─ {group_name}:")
                    for exp_id in exp_ids:
                        lines.append(f"      • {exp_id}")
                else:
                    lines.append(f"   └─ {group_name}: (empty)")

    # Add summary statistics
    if include_stats:
        total_training = sum(len(exps) for exps in all_experiments['training_data'].values())
        total_inference = sum(len(exps) for exps in all_experiments['inference_data'].values())
        lines.append(
            f"\n📊 Summary: {total_training} training experiment(s), "
            f"{total_inference} inference experiment(s)"
        )

    return '\n'.join(lines)

def get_full_experiment_config(
    exp_id: str,
    validated_configs: Optional[ValidatedConfigs] = None
) -> tuple[dict[str, Any], str, str]:
    """
    Builds complete experiment configuration by merging templates and metadata.

    Uses validated Pydantic objects to ensure type safety and automatic validation,
    then merges with:
        - Project constants from experiments.yaml
        - Class definitions from class_definitions.yaml (elemental fractions)
        - Raw/processed config templates from config.yaml
        - Metadata templates from config.yaml

    Args:
        exp_id: Experiment identifier (e.g., '2024-06-29_Dy100')
        validated_configs: Pre-validated configs from verify_experiment_references() (optional).
                          If None, validates all configs automatically.

    Returns:
        Tuple containing:
            - full_config: Merged configuration dictionary with:
                - Training: composition_label, elemental_fractions, beam_current, etc.
                - Inference: possible_composition_labels, mixture_priors (expanded),
                  class_definitions (all elemental fractions), beam_current, etc.
            - dataset_type: 'training_data' or 'inference_data'
            - group_name: Group identifier (e.g., 'physical_single_particle')

    Raises:
        ValueError: If experiment ID not found in experiments.yaml
        ValidationError: If experiments.yaml or other configs are invalid

    Note:
        For processing multiple experiments efficiently, call verify_experiment_references()
        once and pass the result to all subsequent calls to this function.

    Example:
        # Simple usage (validates each time)
        config, dtype, group = get_full_experiment_config('2024-06-29_Dy100')

        # Optimized usage (validate once, reuse many times)
        validated = verify_experiment_references()

        for exp_id in all_experiment_ids:
            config, dtype, group = get_full_experiment_config(exp_id, validated)
    """
    if validated_configs is None:
        validated_configs = verify_experiment_references()

    class_defs, pipeline_config, experiments_db = validated_configs

    configs = pipeline_config.definitions
    classes = {cls: cls_obj.model_dump()
               for cls, cls_obj in class_defs.nanoparticle_classes.items()}

    # Compute config reference fields dynamically (same as in verify_experiment_references)
    ref_fields = [
        name for name, field in BaseExperiment.model_fields.items()
        if isinstance(field.json_schema_extra, dict) and field.json_schema_extra.get('is_config_ref')
    ]

    # Search both datasets
    for dataset_type, dataset in [
        ('training_data', experiments_db.get_training_dataset()),
        ('inference_data', experiments_db.get_inference_dataset())
    ]:
        for group_name in dataset.get_group_names():
            if exp_id in dataset.get_experiment_ids(group_name):
                exp_obj = dataset.get_group_experiments(group_name)[exp_id]

                # Build merged config dict from validated Pydantic object
                exp_config = exp_obj.model_dump()

                # Merge project constants
                exp_config.update(experiments_db.project_constants.model_dump())

                # Merge class definitions
                if dataset_type == 'training_data':
                    # Merge single class definition (elemental fractions)
                    exp_config.update(classes[exp_obj.composition_label])
                else:
                    # Inference: Merge all possible class definitions
                    # Note: mixture_priors already expanded by InferenceExperiment.validate_mixture_priors()
                    exp_config['class_definitions'] = {
                        label: classes[label]
                        for label in exp_obj.possible_composition_labels
                    }

                # Resolve config template references dynamically
                for ref_field in ref_fields:
                    template_name = getattr(exp_obj, ref_field)
                    exp_config[ref_field] = configs[template_name]

                return exp_config, dataset_type, group_name

    # Not found in either dataset
    raise ValueError(f"Experiment ID '{exp_id}' not found in experiments.yaml")

def get_experiment_paths(
    exp_id: str,
    validated_configs: Optional[ValidatedConfigs] = None
) -> dict[str, Path]:
    """
    Constructs directory paths for an experiment's data organization.

    Args:
        exp_id: Experiment identifier (e.g., '2024-06-29_Dy100')
        validated_configs: Optional pre-validated configs from verify_experiment_references().
                          If None, validation is performed automatically.

    Returns:
        Dictionary with keys:
            - 'base': Experiment root directory
            - 'raw': Raw data directory
            - 'processed': Processed data directory
            - 'metadata': Metadata directory

    Example:
        For exp_id='2024-06-29_Dy100', returns paths under:
        data/training/physical/single_particle/2024-06-29_Dy100/

    Performance:
        When processing multiple experiments, validate once and reuse:
        >>> validated = verify_experiment_references()
        >>> for exp_id in all_ids:
        ...     paths = get_experiment_paths(exp_id, validated)
    """
    _, dataset_type, group_name = get_full_experiment_config(exp_id, validated_configs)
    
    # Example: split 'physical_single_particle' into ['physical', 'single_particle']
    path_components = group_name.split('_', 1)

    # Example: data/training/physical/single_particle/2024-06-29_Dy100/
    base_path = (
        PROJECT_ROOT / 'data' / dataset_type.replace('_data', '') / Path(*path_components) / exp_id
    )
    
    return {
        'base': base_path,
        'raw': base_path / 'raw',
        'processed': base_path / 'processed',
        'metadata': base_path / 'metadata'
    }

def get_metadata_filepath(
    exp_id: str,
    metadata_type: str,
    validated_configs: Optional[ValidatedConfigs] = None
) -> Path:
    """
    Constructs path to a metadata file using template substitution.

    Args:
        exp_id: Experiment identifier (e.g., '2024-06-29_Dy100')
        metadata_type: Template key from metadata_templates config (e.g., 'regions')
        validated_configs: Optional pre-validated configs from verify_experiment_references().
                          If None, validation is performed automatically.

    Returns:
        Full path to the metadata file.

    Example:
        For get_metadata_filepath('2024-06-29_Dy100', 'regions'):
        .../metadata/2024-06-29_Dy100_regions.csv
    """
    full_config, _, _ = get_full_experiment_config(exp_id, validated_configs)
    filename_template = full_config['metadata_templates'][metadata_type]
    final_filename = filename_template.format(id=exp_id)
    paths = get_experiment_paths(exp_id, validated_configs)
    return paths['metadata'] / final_filename

def get_raw_filepaths(
    exp_id: str,
    region_id: str,
    validated_configs: Optional[ValidatedConfigs] = None
) -> Optional[dict[str, Path]]:
    """
    Assembles file path patterns for all raw acquisition channels in a region.

    Args:
        exp_id: Experiment identifier (e.g., '2024-06-29_Dy100')
        region_id: Region/FOV identifier (e.g., 'region_1')
        validated_configs: Optional pre-validated configs from verify_experiment_references().
                          If None, validation is performed automatically.

    Returns:
        Dictionary mapping channel identifiers to glob-pattern paths:
            - Simple channels: {'SEM': Path('.../SEM/SEM_FOV_*')}
            - Nested channels: {'CL_blue': Path('.../CL_SE/CL/blue/593CP_FOV_*')}
        Or None if config is incomplete.

    Side Effects:
        Prints error message to console if raw config is incomplete.

    Example:
        Returns {'SEM': Path('.../SEM/SEM_FOV_*'), 'CL_blue': ...}

    Performance:
        When processing multiple regions/experiments, validate once and reuse:
        >>> validated = verify_experiment_references()
        >>> for exp_id in all_ids:
        ...     for region_id in all_regions:
        ...         fps = get_raw_filepaths(exp_id, region_id, validated)
    """
    try:
        filepaths = {}
        paths = get_experiment_paths(exp_id, validated_configs)
        exp_config, _, _ = get_full_experiment_config(exp_id, validated_configs)

        # Base path for this specific region's raw data
        raw_region_path = paths['raw'] / region_id

        # Get the top-level channel block (acquisitions types) from the config
        raw_channels_config = exp_config['raw_config']['acquisitions']
        
        # Traverse acquisition types (CL_SE, SEM)
        for acq_type, acq_config in raw_channels_config.items():
            acq_path = raw_region_path / acq_type
            
            if 'source_prefix' in acq_config:  # Case 1: Simple acquisition (SEM)
                prefix = acq_config['source_prefix']
                filepaths[acq_type] = acq_path / f"{prefix}*"
            
            else:  # Case 2: Nested acquisition (CL_SE)
                for detector, detector_config in acq_config.items():
                    detector_path = acq_path / detector
                    
                    if 'source_prefix' in detector_config:  # Subcase 2a: Simple detector (SE)
                        prefix = detector_config['source_prefix']
                        filepaths[detector] = detector_path / f"{prefix}*"
                
                    else:  # Subcase 2b: Nested detector (CL)
                        for channel_name, channel_info in detector_config.items():
                            channel_path = detector_path / channel_name
                            prefix = channel_info['source_prefix']
                            key_name = f"{detector}_{channel_name}"  # e.g., 'CL_red'
                            filepaths[key_name] = channel_path / f"{prefix}*"
        return filepaths
    
    except KeyError:
        print(f"\nERROR: Could not find raw config for experiment '{exp_id}'", file=sys.stderr)
        return None
   
def get_processed_filepaths(
    exp_id: str,
    method_name: str,
    region_id: str,
    validated_configs: Optional[ValidatedConfigs] = None
) -> Optional[dict[str, Path]]:
    """
    Assembles exact file paths for all processed outputs of a region and method.

    Args:
        exp_id: Experiment identifier (e.g., '2024-06-29_Dy100')
        method_name: Processing method name (e.g., 'summed_images')
        region_id: Region/FOV identifier (e.g., 'region_1')
        validated_configs: Optional pre-validated configs from verify_experiment_references().
                          If None, validation is performed automatically.

    Returns:
        Dictionary mapping channel identifiers to exact file paths:
            - Simple channels: {'SEM': Path('.../summed_SEM_region_1.tif')}
            - Nested channels: {'CL_blue': Path('.../summed_blue_region_1.tif')}
        Or None if config is incomplete.

    Side Effects:
        Prints error message to console if processed config is incomplete.

    Example:
        Returns {'SEM': Path('.../summed_images/SEM/summed_SEM_region_1.tif'), 'CL_blue': ...}

    Performance:
        When processing multiple regions/experiments, validate once and reuse:
        >>> validated = verify_experiment_references()
        >>> for exp_id in all_ids:
        ...     for region_id in all_regions:
        ...         fps = get_processed_filepaths(exp_id, 'summed_images', region_id, validated)
    """
    try:
        filepaths = {}
        paths = get_experiment_paths(exp_id, validated_configs)
        exp_config, _, _ = get_full_experiment_config(exp_id, validated_configs)

        # Base path for this method's outputs, for this region
        output_region_path = paths['processed'] / method_name / region_id

        # Get the top-level channel block (acquisitions types) from the config
        method_config = exp_config['processed_config'][method_name]
        processed_channels_config = method_config['outputs']

        # Traverse acquisition types (CL_SE, SEM)
        for acq_type, acq_config in processed_channels_config.items():
            acq_path = output_region_path / acq_type

            if 'filename_prefix' in acq_config:  # Case 1: Simple acquisition (SEM)
                prefix = acq_config['filename_prefix']
                filename = f"{prefix}_{region_id}.tif"
                filepaths[acq_type] = acq_path / filename
            
            else:  # Case 2: Nested acquisition (CL_SE)
                for detector, detector_config in acq_config.items():
                    detector_path = acq_path / detector

                    if 'filename_prefix' in detector_config:  # Subcase 2a: Simple detector (SE)
                        prefix = detector_config['filename_prefix']
                        filename = f"{prefix}_{region_id}.tif"
                        filepaths[detector] = detector_path / filename

                    else:  # Subcase 2b: Nested detector (CL)
                        for channel_name, channel_info in detector_config.items():
                            channel_path = detector_path / channel_name
                            prefix = channel_info['filename_prefix']
                            filename = f"{prefix}_{region_id}.tif"
                            key_name = f"{detector}_{channel_name}"
                            filepaths[key_name] = channel_path / filename
        return filepaths
                            
    except KeyError:
        print(
            f"\nERROR: Could not find processed config for method '{method_name}' "
            f"for experiment '{exp_id}'",
            file=sys.stderr
        )
        return None

def get_raw_channel_info(
    exp_id: str,
    acq_type: str,
    detector: Optional[str] = None,
    channel_name: Optional[str] = None,
    validated_configs: Optional[ValidatedConfigs] = None
    ) -> Optional[dict[str, Any]]:
    """
    Gets the configuration dictionary for a raw data channel from the full, merged config.

    Args:
        exp_id: The ID of the experiment.
        acq_type: The top-level acquisition type ('CL_SE' or 'SEM').
        detector: The detector type ('CL' or 'SE'), if applicable.
        channel_name: The specific CL channel name ('blue', 'green', 'red'), if applicable.
        validated_configs: Optional pre-validated configs from verify_experiment_references().
                          If None, validation is performed automatically.

    Returns:
        Dict with channel configuration, or None if path is invalid.

    Side Effects:
        Prints error message to console if channel not found.

    Example:
        # Get info for a specific CL channel
        info = get_raw_channel_info('2024-06-29_Dy100', 'CL_SE', 'CL', 'red')
        # Returns: {'target_dopant': 'Ho', 'source_prefix': '593LP_FOV_'}

        # Get info for SE detector
        info = get_raw_channel_info('2024-06-29_Dy100', 'CL_SE', 'SE', None)
        # Returns: {'target_dopant': None, 'source_prefix': 'SE2_FOV_'}

        # Get info for entire CL_SE acquisition
        info = get_raw_channel_info('2024-06-29_Dy100', 'CL_SE', None, None)
        # Returns: {'CL': {...}, 'SE': {...}}
    """
    try:
        full_config, _, _ = get_full_experiment_config(exp_id, validated_configs)
        
        # Start at the top-level channels block (acquisition types)
        info = full_config['raw_config']['acquisitions'][acq_type]
        
        # Go deeper if a detector is specified
        if detector:
            info = info[detector]
        
        # Go deeper if a channel_name is specified
        if channel_name:
            info = info[channel_name]
            
        return info

    except KeyError:
        path_parts = [part for part in [acq_type, detector, channel_name] if part]
        error_path = "/".join(path_parts)
        print(f"\nERROR: Could not find raw info for {exp_id} at path: '{error_path}'")
        return None
    
def get_processed_channel_info(
    exp_id: str,
    method_name: str,
    acq_type: str,
    detector: Optional[str] = None,
    channel_name: Optional[str] = None,
    validated_configs: Optional[ValidatedConfigs] = None
    ) -> Optional[dict[str, Any]]:
    """
    Gets the configuration dictionary for a processed data channel.

    Args:
        exp_id: The ID of the experiment.
        method_name: The name of the processing step (e.g., 'summed_images').
        acq_type: The top-level acquisition type ('CL_SE' or 'SEM').
        detector: The detector type ('CL' or 'SE'), if applicable.
        channel_name: The specific CL channel name ('blue', 'green', 'red'), if applicable.
        validated_configs: Optional pre-validated configs from verify_experiment_references().
                          If None, validation is performed automatically.

    Returns:
        Dict with channel configuration, or None if path is invalid.

    Side Effects:
        Prints error message to console if channel not found.

    Example:
        # Get info for a specific processed channel
        info = get_processed_channel_info('2024-06-29_Dy100', 'summed_images', 'CL_SE', 'CL', 'blue')
        # Returns: {'filename_prefix': 'summed_blue'}

        # Get info for SE detector
        info = get_processed_channel_info('2024-06-29_Dy100', 'summed_images', 'CL_SE', 'SE')
        # Returns: {'filename_prefix': 'summed_SE'}
    """
    try:
        full_config, _, _ = get_full_experiment_config(exp_id, validated_configs)
        
        # Start at the top-level channels block (acquisition types) for the specified method
        method_config = full_config['processed_config'][method_name]
        info = method_config['outputs'][acq_type]
        
        # Go deeper if a detector is specified
        if detector:
            info = info[detector]
        
        # Go deeper if a channel_name is specified
        if channel_name:
            info = info[channel_name]
            
        return info

    except KeyError:
        path_parts = [part for part in [acq_type, detector, channel_name] if part]
        error_path = "/".join(path_parts)
        print(
            f"\nERROR: Could not find processed info for {exp_id} with method ",
            f"'{method_name}', path: '{error_path}'"
        )
        return None

# ===================================================================
# Example Usage (for testing the module directly)
# ===================================================================

def main() -> None:  # pragma: no cover
    """Runs a demonstration and smoke test for the config_loader module."""

    # Define a list of experiment IDs to test
    test_ids = [
        '2024-06-29_Dy100',  # Training data
        '2025-07-02_6-component-mixture',  # Inference data
        'invalid_experiment_id'  # Invalid ID
    ]
    
    print("\n" + "="*80)
    print(" ** RUNNING CONFIG LOADER DEMO & SMOKE TEST **")
    print("="*80)

    # ===== PART 1: Comprehensive Validation (all files + cross-references) =====
    print("\n" + "="*80)
    print("PART 1: Configuration Validation")
    print("="*80)

    print("\n--- Testing verify_experiment_references() ---")
    print()
    try:
        validated_configs = verify_experiment_references()
    except (ValueError, ValidationError, FileNotFoundError) as e:
        print(f"   ❌ FAILED: {e}")
        return
    
    # ===== PART 2: Experiment Discovery =====
    print("\n" + "="*80)
    print("PART 2: Experiment Discovery")
    print("="*80)

    print("\n--- Testing format_experiments_summary() ---")
    try:
        summary = format_experiments_summary(validated_configs, include_stats=True)
        print(summary)
        print("\n✅ Success! Formatted experiment summary")
    except (ValueError, ValidationError) as e:
        print(f"   ❌ FAILED: {e}")

    print("\n--- Testing get_all_experiments() ---\n")
    try:
        all_exps = get_all_experiments()
        if all_exps and (all_exps.get('training_data') or all_exps.get('inference_data')):
            print("✅ Success! get_all_experiments() returned valid structure")
            # Show just counts (format_experiments_summary already shows details)
            train_count = sum(len(exps) for exps in all_exps.get('training_data', {}).values())
            infer_count = sum(len(exps) for exps in all_exps.get('inference_data', {}).values())
            print(f"   {train_count} training, {infer_count} inference experiment(s)")
        else:
            print("   ❌ FAILED: get_all_experiments() returned an empty or invalid structure.")
    except (ValueError, ValidationError, FileNotFoundError) as e:
        print(f"   ❌ FAILED: {e}")

    # ===== PART 3: Detailed Function Tests =====
    print("\n" + "="*80)
    print("PART 3: Detailed API Function Tests")
    print("(Reusing validated_configs from Part 1 for optimal performance)")
    print("="*80)

    for test_exp_id in test_ids:
        print(f"\n\n--- Testing Experiment ID: '{test_exp_id}' ---")
        
        try:
            # Step 3.1: Get the full, merged configuration
            print("\n1. Testing get_full_experiment_config()...")
            full_config, dataset, group = get_full_experiment_config(test_exp_id, validated_configs)
            if full_config and dataset and group:
                print(f"   ✅ Success! Found in: {dataset} -> {group}")

                # Show composition info (differs between training and inference)
                if 'composition_label' in full_config:  # Training data
                    print(f"     - Found 'composition_label': {full_config['composition_label']}")
                    fractions = full_config.get('elemental_fractions', {})
                    if fractions:
                        print(f"     - Merged 'elemental_fractions': {fractions}")
                elif 'possible_composition_labels' in full_config:  # Inference data
                    labels = full_config['possible_composition_labels']
                    print(f"     - Found 'possible_composition_labels': {len(labels)} classes")
                    print(f"       Classes: {', '.join(labels)}")
                    priors = full_config.get('mixture_priors', {})
                    if priors:
                        print(f"     - Found 'mixture_priors': {len(priors)} entries")

                # Show other merged keys
                print(f"     - Merged 'host_matrix': {full_config.get('host_matrix')}")

                raw_keys = list(full_config.get('raw_config', {}).keys())
                print(f"     - Merged 'raw_config' keys: {raw_keys}")

                proc_keys = list(full_config.get('processed_config', {}).keys())
                print(f"     - Merged 'processed_config' keys: {proc_keys}")

                meta_keys = list(full_config.get('metadata_templates', {}).keys())
                print(f"     - Merged 'metadata_templates' keys: {meta_keys}")

            else:
                # This case should be caught by the exception (is included for robustness)
                raise ValueError("get_full_experiment_config() returned empty or incomplete data.")

            # Step 3.2: Get all directory paths
            print("\n2. Testing get_experiment_paths()...")
            paths = get_experiment_paths(test_exp_id, validated_configs)
            if paths and all(k in paths for k in ['base', 'raw', 'processed', 'metadata']):
                print(f"   ✅ Success! Constructed all required paths under '{paths['base'].name}'.")
                for name, path in paths.items():
                    print(f"     - {name.capitalize()} Path: {path}")
            else:
                raise ValueError("FAILED: get_experiment_paths() did not return all required keys.")

            # Step 3.3: Get specific metadata filepath
            print("\n3. Testing get_metadata_filepath()...")
            test_metadata_type = 'regions'
            regions_csv_path = get_metadata_filepath(
                test_exp_id, 
                metadata_type=test_metadata_type, 
                validated_configs=validated_configs
            )
            if regions_csv_path and regions_csv_path.name.endswith('.csv'):
                print(
                    "   ✅ Success! Assembled metadata filepath (for metadata type "
                    f"'{test_metadata_type}'): {regions_csv_path}"
                )
            else:
                raise ValueError(
                    f"get_metadata_filepath() returned an invalid path (for metadata type "
                    f"'{test_metadata_type}')."
                )

            # Step 3.4: Get dictionary of raw file path patterns
            print("\n4. Testing get_raw_filepaths()...")
            test_region_id = 'region_1'
            raw_filepaths = get_raw_filepaths(
                test_exp_id, region_id=test_region_id, validated_configs=validated_configs
                )
            if raw_filepaths:
                print(
                    f"   ✅ Success! Assembled {len(raw_filepaths)} raw filepath patterns ",
                    f"(for region '{test_region_id}'):"
                )
                for key, path_pattern in raw_filepaths.items():
                    print(f"     - {key}: {path_pattern}")
            else:
                print(
                    "   ⚠️ WARNING: get_raw_filepaths() returned an empty dictionary ",
                    f"(for region '{test_region_id}')."
                )

            # Step 3.5: Get dictionary of processed file paths
            print("\n5. Testing get_processed_filepaths()...")
            test_method_name = 'summed_images'
            processed_filepaths = get_processed_filepaths(
                test_exp_id, 
                method_name=test_method_name, 
                region_id=test_region_id, 
                validated_configs=validated_configs
            )
            if processed_filepaths:
                print(
                    f"   ✅ Success! Assembled {len(processed_filepaths)} processed filepaths ",
                    f"(for method '{test_method_name}', region '{test_region_id}'):"
                )
                for key, path in processed_filepaths.items():
                    print(f"     - {key}: {path}")
            else:
                print(
                    "   ⚠️ WARNING: get_processed_filepaths() returned an empty dictionary ",
                    f"(for method '{test_method_name}', region '{test_region_id}')."
                )

            # Step 3.6: Get info for specific raw channel
            print("\n6. Testing get_raw_channel_info()...")
            test_acq_type_raw = 'CL_SE'
            test_detector_raw = 'CL'
            test_channel_name_raw = 'red'
            channel_path_raw = f"{test_acq_type_raw}/{test_detector_raw}"
            if test_channel_name_raw:
                channel_path_raw += f"/{test_channel_name_raw}"
            raw_info = get_raw_channel_info(
                test_exp_id,
                acq_type=test_acq_type_raw,
                detector=test_detector_raw,
                channel_name=test_channel_name_raw,
                validated_configs=validated_configs)
            if raw_info and 'target_dopant' in raw_info and 'source_prefix' in raw_info:
                print(f"   ✅ Success! Found raw info (for channel '{channel_path_raw}'): {raw_info}")
            else:
                print(
                    "   ⚠️ WARNING: get_raw_channel_info() returned invalid or incomplete info ",
                    f"(for channel '{channel_path_raw}')."
                )

            # Step 3.7: Get info for specific processed channel
            print("\n7. Testing get_processed_channel_info()...")
            test_acq_type_proc = 'CL_SE'
            test_detector_proc = 'SE'
            test_channel_name_proc = None
            channel_path_proc = f"{test_acq_type_proc}/{test_detector_proc}"
            if test_channel_name_proc:
                channel_path_proc += f"/{test_channel_name_proc}"
            processed_info = get_processed_channel_info(
                test_exp_id,
                method_name=test_method_name,
                acq_type=test_acq_type_proc,
                detector=test_detector_proc,
                channel_name=test_channel_name_proc,
                validated_configs=validated_configs)
            if processed_info and 'filename_prefix' in processed_info:
                print(
                    f"   ✅ Success! Found processed info (for method '{test_method_name}', ",
                    f"channel '{channel_path_proc}'): {processed_info}"
                )
            else:
                print(
                    "   ⚠️ WARNING: get_processed_channel_info() returned invalid or incomplete info ",
                    f"(for method '{test_method_name}', channel '{channel_path_proc}')."
                )

        except (ValueError, ValidationError, FileNotFoundError) as e:
            print(f"\n   ❌ An error occurred for invalid ID '{test_exp_id}':")
            print(f"      > {e}")
        
    print("\n" + "="*80)
    print("--- DEMO COMPLETE ---")
    print("="*80)

if __name__ == "__main__":
    main()
    