"""
Pydantic schemas for validating YAML configuration files.

This module provides type-safe validation schemas for:
- class_definitions.yaml: Nanoparticle composition database
- experiments.yaml: Experiment database with training and inference data
- config.yaml: Reusable configuration templates

Usage:
    from src.data.schemas import ClassDefinitions, PipelineConfig, ExperimentsDatabase

    # Load and validate class definitions
    class_defs = ClassDefinitions.model_validate(yaml_dict)

    # Load and validate pipeline config
    pipeline_config = PipelineConfig.model_validate(yaml_dict)

    # Load and validate experiments
    experiments_db = ExperimentsDatabase.model_validate(yaml_dict)
"""

import math
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, PrivateAttr, model_validator

# Tolerance for math.isclose() operations
TOLERANCE = 1e-6


# ===================================================================
# Schemas for class_definitions.yaml
# ===================================================================

class ElementalFractions(BaseModel):
    """
    Elemental composition fractions for a nanoparticle class in
    class_definitions.yaml.
    
    Dynamically validates:
        - All required elements are present 
        - There is no extra element
        - Fractions are entered, either as float or int, and are in range [0, 1]
        - Fractions sum to 1.0 (within tolerance)
    """
    Tb: float | int = Field(..., ge=0, le=1, description="Terbium fraction")
    Dy: float | int = Field(..., ge=0, le=1, description="Dysprosium fraction")
    Ho: float | int = Field(..., ge=0, le=1, description="Holmium fraction")
    Gd: float | int = Field(..., ge=0, le=1, description="Gadolinium fraction")
    Y: float | int = Field(..., ge=0, le=1, description="Yttrium fraction")

    model_config = {"extra": "forbid"}

    @model_validator(mode='after')
    def validate_sum(self) -> 'ElementalFractions':
        """Ensure elemental fractions sum to 1.0."""
        total = self.Tb + self.Dy + self.Ho + self.Gd + self.Y

        if not math.isclose(total, 1.0, abs_tol=TOLERANCE):
            raise ValueError(
                f"Elemental fractions must sum to 1.0 (got {total:.10f}). "
                f"Fractions: Tb={self.Tb}, Dy={self.Dy}, Ho={self.Ho}, "
                f"Gd={self.Gd}, Y={self.Y}"
            )

        return self

class NanoparticleClass(BaseModel):
    """Definition of a nanoparticle composition class."""
    elemental_fractions: ElementalFractions = Field(
        ..., description="Fractional composition of rare-earth elements"
    )

class ClassDefinitions(BaseModel):
    """
    Validates the complete class_definitions.yaml file structure.

    This schema uses Pydantic's automatic validation to ensure the YAML file
    contains properly formatted nanoparticle class definitions. All nested
    validation happens automatically during Pydantic's validation phase.

    Validates automatically:
        - Top-level 'nanoparticle_classes' key exists and is not empty
        - No extra top-level keys are present
        - Each class is a valid NanoparticleClass (contains 'elemental_fractions')
        - All elemental fractions pass ElementalFractions validation
          (5 elements present, no extras, in range [0,1], sum to 1.0)

    Usage:
        import yaml
        from src.data.schemas import ClassDefinitions

        with open('class_definitions.yaml') as f:
            yaml_dict = yaml.safe_load(f)

        # Validates entire file structure automatically
        class_defs = ClassDefinitions.model_validate(yaml_dict)

        # Access validated data directly
        labels = class_defs.get_class_labels()
        dy100 = class_defs.get_class('Dy100')
        fractions = class_defs.get_elemental_fractions('Dy100')

    Raises:
        ValidationError: If YAML structure is invalid or validation fails
        KeyError: If requesting a class label that doesn't exist
    """
    nanoparticle_classes: dict[str, NanoparticleClass] = Field(
        ..., description="Dictionary of all nanoparticle classes (validated automatically)"
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode='after')
    def validate_not_empty(self) -> 'ClassDefinitions':
        """Ensure nanoparticle_classes is not empty."""
        if not self.nanoparticle_classes:
            raise ValueError(
                "nanoparticle_classes cannot be empty. "
                "At least one nanoparticle class must be defined."
            )
        return self

    def get_class_labels(self) -> list[str]:
        """Get list of all defined class labels."""
        return list(self.nanoparticle_classes.keys())

    def get_class(self, class_label: str) -> NanoparticleClass:
        """Get a specific nanoparticle class definition."""
        if class_label not in self.nanoparticle_classes:
            raise KeyError(f"Class '{class_label}' not found in class_definitions.yaml")
        return self.nanoparticle_classes[class_label]

    def get_elemental_fractions(self, class_label: str) -> ElementalFractions:
        """Get elemental fractions for a specific class."""
        return self.get_class(class_label).elemental_fractions
    

# ===================================================================
# Schemas for config.yaml
# ===================================================================

class PipelineConfig(BaseModel):
    """
    Schema for validating config.yaml structure.

    This file uses YAML anchors/aliases extensively, so validation focuses on:
    - Required template names exist
    - Hardware-constant acquisition types and detectors are present
    - All raw data channels have required configuration (source_prefix)
    - Metadata templates have proper placeholders

    The validation is flexible enough to allow variable number of CL channels
    while enforcing the physical constraints of the imaging system.
    """
    definitions: dict[str, Any] = Field(
        ..., description="Configuration templates"
    )

    @model_validator(mode='after')
    def validate_template_names(self) -> 'PipelineConfig':
        """Ensure expected template names exist in definitions."""
        expected_templates = {
            'default_raw_config',
            'default_processed_config',
            'default_metadata_templates'
        }

        missing = expected_templates - set(self.definitions.keys())
        if missing:
            raise ValueError(
                f"config.yaml missing expected template(s) in definitions: {sorted(missing)}"
            )

        return self

    @model_validator(mode='after')
    def validate_raw_config_structure(self) -> 'PipelineConfig':
        """
        Validate default_raw_config matches physical instrument setup.

        The imaging system has two acquisition types (hardware constants):
        - CL_SE: Combined cathodoluminescence (CL) and secondary electron (SE) imaging
        - SEM: Standalone higher-resolution SEM imaging

        CL can have variable number of channels (typically 3: blue/green/red).
        """
        raw_config = self.definitions.get('default_raw_config', {})
        acquisitions = raw_config.get('acquisitions', {})

        # Validate required acquisition types (hardware constant)
        required_acq_types = {'CL_SE', 'SEM'}
        missing = required_acq_types - set(acquisitions.keys())
        if missing:
            raise ValueError(
                f"default_raw_config missing required acquisition type(s): {sorted(missing)}. "
                f"These are hardware constants for the imaging system."
            )

        # Validate CL_SE structure (hardware constant)
        cl_se = acquisitions.get('CL_SE', {})
        required_detectors = {'CL', 'SE'}
        missing_detectors = required_detectors - set(cl_se.keys())
        if missing_detectors:
            raise ValueError(
                f"CL_SE acquisition missing detector(s): {sorted(missing_detectors)}. "
                f"The system requires both CL and SE detectors."
            )

        # Validate CL has at least one channel (flexible number)
        cl = cl_se.get('CL', {})
        if not cl or not isinstance(cl, dict):
            raise ValueError(
                "CL detector must have at least one channel defined "
                "(e.g., blue, green, red, or wavelength-based names)."
            )

        # Validate all CL channels have source_prefix
        for channel_name, channel_config in cl.items():
            if not isinstance(channel_config, dict) or 'source_prefix' not in channel_config:
                raise ValueError(
                    f"CL channel '{channel_name}' must have 'source_prefix' defined."
                )

        # Validate SE has source_prefix
        if not isinstance(cl_se.get('SE'), dict) or 'source_prefix' not in cl_se['SE']:
            raise ValueError("SE detector must have 'source_prefix' defined.")

        # Validate SEM has source_prefix
        if not isinstance(acquisitions.get('SEM'), dict) or 'source_prefix' not in acquisitions['SEM']:
            raise ValueError("SEM acquisition must have 'source_prefix' defined.")

        return self

    @model_validator(mode='after')
    def validate_metadata_templates(self) -> 'PipelineConfig':
        """Ensure metadata templates contain {id} placeholder for experiment ID substitution."""
        templates = self.definitions.get('default_metadata_templates', {})

        for template_name, template_str in templates.items():
            if not isinstance(template_str, str):
                raise ValueError(
                    f"Metadata template '{template_name}' must be a string, got {type(template_str)}"
                )
            if '{id}' not in template_str:
                raise ValueError(
                    f"Metadata template '{template_name}' missing {{id}} placeholder. Got: '{template_str}'"
                )

        return self
    

# ===================================================================
# Schemas for Experiments.yaml
# ===================================================================

class ProjectConstants(BaseModel):
    """Project-wide experimental constants."""
    substrate: str = Field(..., description="Substrate material for the nanoparticles")
    host_matrix: str = Field(..., description="Host elements matrix for the nanoparticles")
    instrument_model: str = Field(..., description="Electron microscope model")
    CL_detection_system: dict[str, dict[str, Any]] = Field(..., description="Cathodoluminescence detection system")
    SE_detection_system: dict[str, str] = Field(..., description="Secondary electron detection system")

class BeamCurrent(BaseModel):
    """Electron beam current information."""
    estimate: int | float = Field(..., gt=0, description="Beam current estimate (must be positive)")
    unit: Literal['pA', 'nA', 'A'] = Field(..., description="Beam current unit (either 'pA', 'nA', or 'A')")
    type: Literal['nominal', 'measured'] = Field(..., description="Estimate type (either 'nominal' or 'measured')")
    notes: str = Field(default="", description="Additional notes about beam current")

class BaseExperiment(BaseModel):
    """Base class for fields shared between all experiment."""
    beam_current: BeamCurrent = Field(..., description="Electron beam current information")
    raw_config: str = Field(
        ...,
        description="Reference to raw config template in config.yaml (must be exact match)",
        json_schema_extra={'is_config_ref': True}
    )
    processed_config: str = Field(
        ...,
        description="Reference to processed config template in config.yaml (must be exact match)",
        json_schema_extra={'is_config_ref': True}
    )
    metadata_templates: str = Field(
        ...,
        description="Reference to metadata templates in config.yaml (must be exact match)",
        json_schema_extra={'is_config_ref': True}
    )

class TrainingExperiment(BaseExperiment):
    """
    Experiment in training dataset (nanoparticles with known composition(s)).

    **Current Scope (<=v1.0)**: Single-particle datasets only (physical_single_particle group).
    Each experiment's region contains a single particle of known composition (from chemical synthesis).

    **Planned Extension (>=v2.0)**: Simulated training data will be added:
    - simulated_single_particle: 
      - Each region contains a single particle of known composition (as per simulation specs).
      - Same schema as physical (composition_label: str).
    - simulated_multi_particle: 
      - Each region contains multiple particles, each of which the composition is known (as per simulation specs).
      - Different schema with per-particle ground truth:
        - Will use composition_labels: List[str] field instead of composition_label
        - Actual per-particle labels stored in metadata CSV: {experiment_id}_particle_labels.csv
        - CSV columns: region_id, particle_id, composition_label, centroid_x, centroid_y, ...
        - Schema type will be inferred from group name (_single_particle vs _multi_particle suffix)

    Note: Cross-referencing between composition_label in experiments.yaml and the
    classes defined in class_definitions.yaml is validated in config_loader.py.
    """
    composition_label: str = Field(
        ..., min_length=1, description="Known composition class (e.g., 'Dy100', 'Tb50Ho50')"
    )

class InferenceExperiment(BaseExperiment):
    """
    Experiment in inference dataset (mixture of nanoparticles, with only prior knowledge
    of composition).

    Note: Cross-referencing between possible_composition_labels in experiments.yaml 
    and the classes defined in class_definitions.yaml is validated in config_loader.py.
    """
    possible_composition_labels: list[str] = Field(
        ..., min_length=1, description="List of possible nanoparticle classes in the mixture"
    )

    mixture_priors: str | dict[str, int | float] = Field(
        ..., 
        description=(
            "Prior probabilities for each composition class. "
            "Can be either:\n"
            "  - A strategy string: 'equal_split' (auto-calculates 1/N for each class)\n"
            "  - An explicit dict: {'Dy100': 0.5, 'Ho100': 0.5} (must sum to 1)"
        )
    )

    @model_validator(mode='after')
    def validate_mixture_priors(self) -> 'InferenceExperiment':
        """
        Validate and expand mixture_priors values, and validate its consistency with
        possible_composition_labels.

        If mixture_priors is a strategy string, expand it to a dict before validating
        the dict values.
        
        Checks:
            - All priors are in range [0, 1]
            - Priors sum to 1.0 (within tolerance)
            - Keys match possible_composition_labels exactly
        """
        # Case 1: Strategy string
        if isinstance(self.mixture_priors, str):
            strategy = self.mixture_priors.lower()
            
            if strategy == 'equal_split':
                n_classes = len(self.possible_composition_labels)
                equal_weight = 1.0 / n_classes
                self.mixture_priors = {
                    label: equal_weight 
                    for label in self.possible_composition_labels
                }
            else:
                raise ValueError(
                    f"Unknown mixture_priors strategy: '{self.mixture_priors}'. "
                    f"Supported strategies: 'equal_split'"
                )

        # Case 2: Explicit dict (full validation)
        elif isinstance(self.mixture_priors, dict):
            # Validate each prior value
            for label, prior in self.mixture_priors.items():
                if not isinstance(prior, (int, float)):
                    raise TypeError(f"Prior for '{label}' must be a number, got {type(prior)}")
                if not (0 <= prior <= 1):
                    raise ValueError(f"Prior for '{label}' must be in range [0, 1], got {prior}")
                
            # Check priors sum to 1.0
            priors_sum = sum(self.mixture_priors.values())
            if not math.isclose(priors_sum, 1.0, abs_tol=TOLERANCE):
                raise ValueError(
                    f"Mixture priors must sum to 1.0 (got {priors_sum:.10f}). "
                    f"Current priors: {self.mixture_priors}"
                )

            # Check that all labels have priors
            labels_set = set(self.possible_composition_labels)
            priors_set = set(self.mixture_priors.keys())

            if labels_set != priors_set:
                missing_in_priors = labels_set - priors_set
                extra_in_priors = priors_set - labels_set
                error_parts = []
                if missing_in_priors:
                    error_parts.append(
                        f"Labels missing in mixture_priors: {sorted(missing_in_priors)}"
                    )
                if extra_in_priors:
                    error_parts.append(
                        f"Extra keys in mixture_priors: {sorted(extra_in_priors)}"
                    )
                raise ValueError("; ".join(error_parts))

        else:
            raise TypeError(
                f"mixture_priors must be str or dict, got {type(self.mixture_priors)}"
            )

        return self

class ExperimentDataset(BaseModel):
    """Container for training_data or inference_data."""
    model_config = {"extra": "allow"}  # Any group can be added

    _dataset_type: str = PrivateAttr(default=None)
    _validated_groups: dict[str, dict[str, BaseExperiment]] = PrivateAttr(default_factory=dict)
    
    def __init__(self, dataset_type: str, **data):
        """
        Initialize with dataset type and dynamic group data.
        
        Args:
            dataset_type: Either 'training_data' or 'inference_data'
            **data: Dynamic group names with their experiment dicts
        """
        super().__init__(**data)
        self._dataset_type = dataset_type
        self._validate_and_parse_experiments(data)   

    def _validate_and_parse_experiments(self, data: dict[str, Any]) -> None:
        """
        Validate experiments match dataset type and create validated instances.

        Args:
            data: dict of {group_name: {exp_id: exp_config}}
        
        Raises:
            ValueError: If dataset_type is invalid, or experiment structure is incorrect
            ValidationError: If individual experiments fail Pydantic validation
        """
        if self._dataset_type not in ['training_data', 'inference_data']:
            raise ValueError(
                f"dataset_type must be 'training_data' or 'inference_data', "
                f"got '{self._dataset_type}'"
            )
        
        validated_groups = {}

        for group_name, exp_dict in data.items():
            if not isinstance(exp_dict, dict):
                raise ValueError(
                    f"Group '{group_name}' must be a dict, got {type(exp_dict)}"
                )
            
            experiments = {}
            for exp_id, exp_config in exp_dict.items():
                if not isinstance(exp_config, dict):
                    raise ValueError(
                        f"Experiment '{exp_id}' in group '{group_name}' must be a dict, "
                        f"got {type(exp_config)}"
                    )
                
                if self._dataset_type == "training_data":
                    experiments[exp_id] = TrainingExperiment.model_validate(exp_config)
                elif self._dataset_type == "inference_data":
                    experiments[exp_id] = InferenceExperiment.model_validate(exp_config)
            
            validated_groups[group_name] = experiments
        
        self._validated_groups = validated_groups

    def get_group_names(self) -> list[str]:
        """Get list of group names in this dataset."""
        return list(self._validated_groups.keys())

    def get_group_experiments(self, group_name: str) -> dict[str, BaseExperiment]:
        """Get all experiments in a group."""
        return self._validated_groups[group_name]

    def get_experiment_ids(self, group_name: str) -> list[str]:
        """Get list of experiment IDs in a specific group."""
        return list(self._validated_groups[group_name].keys())

class ExperimentsDatabase(BaseModel):
    """Complete experiments.yaml structure."""
    project_constants: ProjectConstants = Field(
        ..., description="Project-wide experimental constants"
    )
    training_data: dict[str, dict[str, dict[str, Any]]] = Field(
        ..., description="Experiments for training organized by groups"
    )
    inference_data: dict[str, dict[str, dict[str, Any]]] = Field(
        ..., description="Experiments for inference organized by groups"
    )

    _training_dataset: Optional[ExperimentDataset] = None
    _inference_dataset: Optional[ExperimentDataset] = None

    @model_validator(mode='after')
    def parse_datasets(self) -> 'ExperimentsDatabase':
        """Parse and validate training and inference datasets."""
        self._training_dataset = ExperimentDataset(
            dataset_type="training_data", **self.training_data
        )
        self._inference_dataset = ExperimentDataset(
            dataset_type="inference_data", **self.inference_data
        )
        return self

    def get_training_dataset(self) -> ExperimentDataset:
        """Get validated training dataset."""
        return self._training_dataset

    def get_inference_dataset(self) -> ExperimentDataset:
        """Get validated inference dataset."""
        return self._inference_dataset

    def get_all_experiments(self) -> dict[str, dict[str, list[str]]]:
        """
        Get all experiment IDs organized by dataset type and group.

        Returns:
            dict with structure:
            {
                'training_data': {
                    'group_name': ['exp_id1', 'exp_id2', ...],
                    ...
                },
                'inference_data': {
                    'group_name': ['exp_id1', 'exp_id2', ...],
                    ...
                }
            }
        """
        experiments_tree = {'training_data': {}, 'inference_data': {}}

        for group_name in self._training_dataset.get_group_names():
            experiments_tree['training_data'][group_name] = (
                self._training_dataset.get_experiment_ids(group_name)
            )

        for group_name in self._inference_dataset.get_group_names():
            experiments_tree['inference_data'][group_name] = (
                self._inference_dataset.get_experiment_ids(group_name)
            )

        return experiments_tree
