"""
Test suite for the src.data.schemas module.

This suite tests Pydantic validation logic for configuration schemas,
including class definitions, experiments, and pipeline config.
"""

import copy

import pytest
from pydantic import ValidationError

from src.data.schemas import (
    BeamCurrent,
    ClassDefinitions,
    ElementalFractions,
    ExperimentDataset,
    ExperimentsDatabase,
    InferenceExperiment,
    NanoparticleClass,
    PipelineConfig,
    TrainingExperiment,
)

# Note: In parametrized tests, the first parameter (_test_name) is used by pytest
# to generate test IDs but is not accessed in the test body. This is intentional.


# ===============================================================================================
#    Test schemas for class_definitions.yaml
# ===============================================================================================

# ===================================================================
# ElementalFractions Tests
# ===================================================================

@pytest.mark.parametrize(
    "_test_name, mock_fractions, should_pass, error_match",
    [
        # Valid cases
        ("Valid_Integer", {'Tb': 0, 'Dy': 1, 'Ho': 0.0, 'Gd': 0.0, 'Y': 0.0}, True, None),
        ("Valid_Mixed", {'Tb': 0.2, 'Dy': 0.3, 'Ho': 0.5, 'Gd': 0.0, 'Y': 0.0}, True, None),
        ("Valid_EqualSplit", {'Tb': 0.2, 'Dy': 0.2, 'Ho': 0.2, 'Gd': 0.2, 'Y': 0.2}, True, None),
        ("Valid_OtherOrder", {'Dy': 0.3, 'Tb': 0.2, 'Ho': 0.5, 'Gd': 0.0, 'Y': 0.0}, True, None),

        # Field errors
        ("MissingElement", {'Tb': 0.0, 'Dy': 1.0, 'Ho': 0.0, 'Gd': 0.0}, False, "Field required"),
        ("ExtraElement", {'Tb': 0.0, 'Dy': 1.0, 'Ho': 0.0, 'Gd': 0.0, 'Y': 0.0, 'He': 0.0}, False, "Extra inputs are not permitted"),

        # Range validation errors
        ("NegativeValue", {'Tb': -0.1, 'Dy': 0.6, 'Ho': 0.5, 'Gd': 0.0, 'Y': 0.0}, False, "greater than or equal to"),
        ("ValueOver1", {'Tb': 0.0, 'Dy': 1.2, 'Ho': -0.2, 'Gd': 0.0, 'Y': 0.0}, False, "less than or equal to"),

        # Sum validation errors
        ("SumOver", {'Tb': 0.0, 'Dy': 0.8, 'Ho': 0.3, 'Gd': 0.0, 'Y': 0.0}, False, "must sum to 1.0"),
        ("SumBelow", {'Tb': 0.0, 'Dy': 0.8, 'Ho': 0.1, 'Gd': 0.0, 'Y': 0.0}, False, "must sum to 1.0"),

        # Type validation errors
        ("StringValue", {'Tb': 0.0, 'Dy': 'not-a-number', 'Ho': 1.0, 'Gd': 0.0, 'Y': 0.0}, False, "Input should be a valid number"),
        ("NoneValue", {'Tb': None, 'Dy': 0.5, 'Ho': 0.5, 'Gd': 0.0, 'Y': 0.0}, False, "Input should be a valid number"),
    ]
)

def test_elemental_fractions_parametrized(_test_name, mock_fractions, should_pass, error_match):
    """Test ElementalFractions validation with various valid and invalid inputs."""
    if should_pass:
        fractions = ElementalFractions(**mock_fractions)
        assert fractions.Tb == mock_fractions['Tb']
    else:
        with pytest.raises(ValidationError, match=error_match):
            ElementalFractions(**mock_fractions)

# ===================================================================
# NanoparticleClass Tests
# ===================================================================

@pytest.mark.parametrize(
    "_test_name, mock_classes, should_pass, error_match",
    [
        # Valid cases
        ("Valid", {'elemental_fractions': {'Tb': 0.0, 'Dy': 1.0, 'Ho': 0.0, 'Gd': 0.0, 'Y': 0.0}}, True, None),

        # Structure errors
        ("WrongType", {'elemental_fractions': "not a dict"}, False, "Input should be a valid dictionary"),
        ("EmptyDict", {'elemental_fractions': {}}, False, "Field required"),
        ("MissingFractions", {'some_other_key': {'Tb': 0.0, 'Dy': 1.0, 'Ho': 0.0, 'Gd': 0.0, 'Y': 0.0}}, False, "Field required"),

        # Propagation test
        ("InvalidClass_MissingElement", {'elemental_fractions': {'Tb': 0.0, 'Dy': 1.0, 'Ho': 0.0, 'Gd': 0.0}}, False, "Field required"),
    ]
)

def test_nanoparticle_class_parametrized(_test_name, mock_classes, should_pass, error_match):
    """Test NanoparticleClass validation with various valid and invalid inputs."""
    if should_pass:
        nanoparticle_class = NanoparticleClass(**mock_classes)
        assert isinstance(nanoparticle_class.elemental_fractions, ElementalFractions)
    else:
        with pytest.raises(ValidationError, match=error_match):
            NanoparticleClass(**mock_classes)

# ===================================================================
# ClassDefinitions Tests
# ===================================================================

@pytest.mark.parametrize(
    "_test_name, classes_data, should_pass, error_match",
    [
        # Valid cases
        ("Valid_SingleClass", {'Dy100': {'elemental_fractions': {'Tb': 0.0, 'Dy': 1.0, 'Ho': 0.0, 'Gd': 0.0, 'Y': 0.0}}}, True, None),
        ("Valid_MultipleClasses", 
            {
                'Dy100': {'elemental_fractions': {'Tb': 0.0, 'Dy': 1.0, 'Ho': 0.0, 'Gd': 0.0, 'Y': 0.0}},
                'Ho100': {'elemental_fractions': {'Tb': 0.0, 'Dy': 0.0, 'Ho': 1.0, 'Gd': 0.0, 'Y': 0.0}}
            },
            True,
            None
        ),

        # Structure errors
        ("Empty", {}, False, "cannot be empty"),

        # Propagation test
        (
            "MissingElementalFractions",
            {'Dy100': {'some_other_key': {'Tb': 0.0, 'Dy': 1.0, 'Ho': 0.0, 'Gd': 0.0, 'Y': 0.0}}},
            False, 
            "Field required"
        ),
    ]
)

def test_class_definitions_parametrized(_test_name, classes_data, should_pass, error_match):
    """Test ClassDefinitions validation with various valid and invalid inputs."""
    if should_pass:
        class_defs = ClassDefinitions(nanoparticle_classes=classes_data)
        for label in classes_data.keys():
            assert label in class_defs.get_class_labels()
    else:
        with pytest.raises(ValidationError, match=error_match):
            ClassDefinitions(nanoparticle_classes=classes_data)

def test_class_definitions_missing_top_level_key():
    """Test ClassDefinitions validation with missing top-level key."""
    yaml_dict = {'some_other_key': {'Dy100': {'elemental_fractions': {...}}}}
    with pytest.raises(ValidationError, match="Field required"):
        ClassDefinitions.model_validate(yaml_dict)


# ===============================================================================================
#    Test schemas for config.yaml
# ===============================================================================================

# ===================================================================
# PipelineConfig Tests
# ===================================================================

@pytest.fixture
def mock_valid_pipeline_config() -> dict:
    """Provides a fixture for a valid PipelineConfig definitions dict."""
    return {
        'default_raw_config': {
            'acquisitions': {
                'CL_SE': {
                    'CL': {
                        'blue': {'source_prefix': 'CP_'},
                        'green': {'source_prefix': 'SP_'},
                        'red': {'source_prefix': 'LP_'}
                    },
                    'SE': {'source_prefix': 'SE2_'}
                },
                'SEM': {'source_prefix': 'SEM_'}
            }
        },
        'default_processed_config': {
            'summed_images': {
                'outputs': {
                    'CL_SE': {
                        'CL': {'red': {'filename_prefix': 'summed_LP'}},
                        'SE': {'filename_prefix': 'summed_SE2'}
                    }
                }
            }
        },
        'default_metadata_templates': {'regions': '{id}_regions.csv'}
    }

# Helper functions for config modifications
def _remove_template_raw(cfg):
    """Remove default_raw_config template."""
    cfg.pop('default_raw_config')
    return cfg

def _remove_template_processed(cfg):
    """Remove default_processed_config template."""
    cfg.pop('default_processed_config')
    return cfg

def _remove_template_metadata(cfg):
    """Remove default_metadata_templates."""
    cfg.pop('default_metadata_templates')
    return cfg

def _remove_acquisition_cl_se(cfg):
    """Remove CL_SE acquisition type."""
    cfg['default_raw_config']['acquisitions'].pop('CL_SE')
    return cfg

def _remove_acquisition_sem(cfg):
    """Remove SEM acquisition type."""
    cfg['default_raw_config']['acquisitions'].pop('SEM')
    return cfg

def _remove_detector_cl(cfg):
    """Remove CL detector from CL_SE."""
    cfg['default_raw_config']['acquisitions']['CL_SE'].pop('CL')
    return cfg

def _remove_detector_se(cfg):
    """Remove SE detector from CL_SE."""
    cfg['default_raw_config']['acquisitions']['CL_SE'].pop('SE')
    return cfg

def _empty_cl_channels(cfg):
    """Set CL channels to empty dict."""
    cfg['default_raw_config']['acquisitions']['CL_SE']['CL'] = {}
    return cfg

def _cl_not_dict(cfg):
    """Set CL to non-dict type."""
    cfg['default_raw_config']['acquisitions']['CL_SE']['CL'] = "not_a_dict"
    return cfg

def _remove_source_prefix_cl_channel(cfg):
    """Remove source_prefix from red CL channel."""
    cfg['default_raw_config']['acquisitions']['CL_SE']['CL']['red'].pop('source_prefix')
    return cfg

def _remove_source_prefix_se(cfg):
    """Remove source_prefix from SE detector."""
    cfg['default_raw_config']['acquisitions']['CL_SE']['SE'].pop('source_prefix')
    return cfg

def _remove_source_prefix_sem(cfg):
    """Remove source_prefix from SEM acquisition."""
    cfg['default_raw_config']['acquisitions']['SEM'].pop('source_prefix')
    return cfg

def _missing_id_placeholder(cfg):
    """Set metadata template without {id} placeholder."""
    cfg['default_metadata_templates']['regions'] = 'regions.csv'
    return cfg

def _metadata_template_not_string(cfg):
    """Set metadata template to non-string type."""
    cfg['default_metadata_templates']['regions'] = 123
    return cfg

@pytest.mark.parametrize(
    "_test_name, mock_config, should_pass, error_match",
    [
        # Valid cases
        ("Valid", lambda cfg: cfg, True, None),

        # Template name missing errors
        ("MissingTemplate_RawConfig", _remove_template_raw, False, "missing expected template"),
        ("MissingTemplate_ProcessedConfig", _remove_template_processed, False, "missing expected template"),
        ("MissingTemplate_MetadataTemplates", _remove_template_metadata, False, "missing expected template"),

        # Raw config structure validation errors - acquisition types
        ("MissingAcquisitionType_CL_SE", _remove_acquisition_cl_se, False, "missing required acquisition type"),
        ("MissingAcquisitionType_SEM", _remove_acquisition_sem, False, "missing required acquisition type"),

        # Raw config structure validation errors - detectors
        ("MissingDetector_CL", _remove_detector_cl, False, "missing detector"),
        ("MissingDetector_SE", _remove_detector_se, False, "missing detector"),

        # Raw config structure validation errors - CL channels
        ("EmptyCLChannels", _empty_cl_channels, False, "at least one channel"),
        ("CLNotDict", _cl_not_dict, False, "at least one channel"),

        # Raw config structure validation errors - source_prefix
        ("MissingSourcePrefix_CLChannel", _remove_source_prefix_cl_channel, False, "must have 'source_prefix'"),
        ("MissingSourcePrefix_SE", _remove_source_prefix_se, False, "SE detector must have 'source_prefix'"),
        ("MissingSourcePrefix_SEM", _remove_source_prefix_sem, False, "SEM acquisition must have 'source_prefix'"),

        # Metadata template validation errors
        ("MissingIDPlaceholder", _missing_id_placeholder, False, "missing {id} placeholder"),
        ("MetadataTemplateNotString", _metadata_template_not_string, False, "must be a string"),
    ]
)

def test_pipeline_config_parametrized(_test_name, mock_config, should_pass, error_match, mock_valid_pipeline_config):
    """Tests validation logic for pipeline configuration including hardware constants and flexible requirements."""
    # Make a deep copy to avoid fixture mutation, then apply modification
    config_copy = copy.deepcopy(mock_valid_pipeline_config)
    modified_config = mock_config(config_copy)

    if should_pass:
        config = PipelineConfig(definitions=modified_config)
        assert config.definitions == modified_config
    else:
        with pytest.raises(ValidationError, match=error_match):
            PipelineConfig(definitions=modified_config)

def test_pipeline_config_missing_top_level_key():
    """Test PipelineConfig validation with missing top-level key."""
    yaml_dict = {'some_other_key': {'some_template': {}}}
    with pytest.raises(ValidationError, match="Field required"):
        PipelineConfig.model_validate(yaml_dict)


# ===============================================================================================
#    Test schemas for experiments.yaml
# ===============================================================================================

# ===================================================================
# BeamCurrent Tests
# ===================================================================

@pytest.mark.parametrize(
    "_test_name, beam_data, should_pass, error_match",
    [
        # Valid cases
        ("Valid_nA_Nominal", {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'}, True, None),
        ("Valid_pA_Nominal", {'estimate': 180.0, 'unit': 'pA', 'type': 'nominal'}, True, None),
        ("Valid_A_Measured", {'estimate': 1.8e-10, 'unit': 'A', 'type': 'measured'}, True, None),
        ("Valid_WithNotes", {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal', 'notes': 'Calibrated'}, True, None),
        ("Valid_IntegerValue", {'estimate': 1, 'unit': 'nA', 'type': 'nominal'}, True, None),

        # Value validation errors
        ("NegativeValue", {'estimate': -0.1, 'unit': 'nA', 'type': 'nominal'}, False, "greater than 0"),
        ("ZeroValue", {'estimate': 0.0, 'unit': 'nA', 'type': 'nominal'}, False, "greater than 0"),
        ("StringValue", {'estimate': "invalid", 'unit': 'nA', 'type': 'nominal'}, False, "Input should be a valid number"),

        # Unit validation errors
        ("InvalidUnit_mA", {'estimate': 0.18, 'unit': 'mA', 'type': 'nominal'}, False, "Input should be 'pA', 'nA' or 'A'"),
        ("InvalidUnit_uA", {'estimate': 0.18, 'unit': 'uA', 'type': 'nominal'}, False, "Input should be 'pA', 'nA' or 'A'"),

        # Type validation errors
        ("InvalidType_estimated", {'estimate': 0.18, 'unit': 'nA', 'type': 'estimated'}, False, "Input should be 'nominal' or 'measured'"),
        ("InvalidType_calculated", {'estimate': 0.18, 'unit': 'nA', 'type': 'calculated'}, False, "Input should be 'nominal' or 'measured'"),

        # Missing fields
        ("MissingEstimate", {'unit': 'nA', 'type': 'nominal'}, False, "Field required"),
        ("MissingUnit", {'estimate': 0.18, 'type': 'nominal'}, False, "Field required"),
        ("MissingType", {'estimate': 0.18, 'unit': 'nA'}, False, "Field required"),
    ]
)
def test_beam_current_parametrized(_test_name, beam_data, should_pass, error_match):
    """Test BeamCurrent validation with various valid and invalid inputs."""
    if should_pass:
        bc = BeamCurrent(**beam_data)
        assert bc.estimate == beam_data['estimate']
        assert bc.unit == beam_data['unit']
        assert bc.type == beam_data['type']
    else:
        with pytest.raises(ValidationError, match=error_match):
            BeamCurrent(**beam_data)


# ===================================================================
# TrainingExperiment Tests
# ===================================================================

@pytest.fixture
def mock_valid_training_experiment() -> dict:
    """Provides a fixture for a valid TrainingExperiment."""
    return {
        'composition_label': 'Dy100',
        'beam_current': {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'},
        'raw_config': 'default_raw_config',
        'processed_config': 'default_processed_config',
        'metadata_templates': 'default_metadata_templates'
    }

# Helper functions for TrainingExperiment modifications
def _remove_composition_label(exp):
    """Remove composition_label."""
    exp.pop('composition_label')
    return exp

def _empty_composition_label(exp):
    """Set composition_label to empty string."""
    exp['composition_label'] = ''
    return exp

def _remove_beam_current(exp):
    """Remove beam_current."""
    exp.pop('beam_current')
    return exp

def _invalid_beam_current(exp):
    """Set beam_current to invalid value."""
    exp['beam_current'] = {'estimate': -0.1, 'unit': 'nA', 'type': 'nominal'}
    return exp

def _remove_raw_config(exp):
    """Remove raw_config reference."""
    exp.pop('raw_config')
    return exp

def _remove_processed_config(exp):
    """Remove processed_config reference."""
    exp.pop('processed_config')
    return exp

def _remove_metadata_templates(exp):
    """Remove metadata_templates reference."""
    exp.pop('metadata_templates')
    return exp

@pytest.mark.parametrize(
    "_test_name, modify_exp, should_pass, error_match",
    [
        # Valid cases
        ("Valid", lambda exp: exp, True, None),
        ("Valid_DifferentLabel", lambda exp: {**exp, 'composition_label': 'Ho100'}, True, None),

        # composition_label validation errors
        ("EmptyLabel", _empty_composition_label, False, "String should have at least 1 character"),
        ("MissingLabel", _remove_composition_label, False, "Field required"),

        # beam_current validation errors
        ("MissingBeamCurrent", _remove_beam_current, False, "Field required"),
        ("InvalidBeamCurrent", _invalid_beam_current, False, "greater than 0"),

        # Config reference validation errors
        ("MissingRawConfig", _remove_raw_config, False, "Field required"),
        ("MissingProcessedConfig", _remove_processed_config, False, "Field required"),
        ("MissingMetadataTemplates", _remove_metadata_templates, False, "Field required"),
    ]
)
def test_training_experiment_parametrized(_test_name, modify_exp, should_pass, error_match, mock_valid_training_experiment):
    """Test TrainingExperiment validation with various valid and invalid inputs."""
    exp_copy = copy.deepcopy(mock_valid_training_experiment)
    modified_exp = modify_exp(exp_copy)

    if should_pass:
        exp = TrainingExperiment(**modified_exp)
        assert exp.composition_label == modified_exp['composition_label']
        assert exp.raw_config == modified_exp['raw_config']
    else:
        with pytest.raises(ValidationError, match=error_match):
            TrainingExperiment(**modified_exp)


# ===================================================================
# InferenceExperiment Tests
# ===================================================================

@pytest.fixture
def mock_valid_inference_experiment() -> dict:
    """Provides a fixture for a valid InferenceExperiment."""
    return {
        'possible_composition_labels': ['Dy100', 'Ho100'],
        'mixture_priors': {'Dy100': 0.3, 'Ho100': 0.7},
        'beam_current': {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'},
        'raw_config': 'default_raw_config',
        'processed_config': 'default_processed_config',
        'metadata_templates': 'default_metadata_templates'
    }

# Helper functions for InferenceExperiment modifications
def _set_equal_split_strategy(exp):
    """Set mixture_priors to 'equal_split' strategy."""
    exp['mixture_priors'] = 'equal_split'
    return exp

def _set_unknown_strategy(exp):
    """Set mixture_priors to unknown strategy."""
    exp['mixture_priors'] = 'weighted_by_magic'
    return exp

def _priors_sum_over_one(exp):
    """Set priors that sum > 1.0."""
    exp['mixture_priors'] = {'Dy100': 0.6, 'Ho100': 0.6}
    return exp

def _priors_sum_under_one(exp):
    """Set priors that sum < 1.0."""
    exp['mixture_priors'] = {'Dy100': 0.3, 'Ho100': 0.3}
    return exp

def _prior_value_negative(exp):
    """Set a negative prior value."""
    exp['mixture_priors'] = {'Dy100': -0.2, 'Ho100': 1.2}
    return exp

def _prior_value_over_one(exp):
    """Set a prior value > 1."""
    exp['mixture_priors'] = {'Dy100': 1.5, 'Ho100': -0.5}
    return exp

def _missing_label_in_priors(exp):
    """Remove a label from mixture_priors."""
    exp['possible_composition_labels'] = ['Dy100', 'Ho100', 'Tb100']
    exp['mixture_priors'] = {'Dy100': 0.5, 'Ho100': 0.5}
    return exp

def _extra_label_in_priors(exp):
    """Add an extra label to mixture_priors."""
    exp['mixture_priors'] = {'Dy100': 0.3, 'Ho100': 0.5, 'Tb100': 0.2}
    return exp

def _non_numeric_prior(exp):
    """Set a non-numeric prior value."""
    exp['mixture_priors'] = {'Dy100': 'high', 'Ho100': 0.7}
    return exp

def _priors_wrong_type(exp):
    """Set mixture_priors to wrong type."""
    exp['mixture_priors'] = 123
    return exp

def _empty_labels_list(exp):
    """Set empty possible_composition_labels."""
    exp['possible_composition_labels'] = []
    return exp

@pytest.mark.parametrize(
    "_test_name, modify_exp, should_pass, error_match",
    [
        # Valid cases - explicit dict
        ("Valid_ExplicitDict", lambda exp: exp, True, None),

        # Valid cases - equal_split strategy
        ("Valid_EqualSplit", _set_equal_split_strategy, True, None),

        # Strategy validation errors
        ("UnknownStrategy", _set_unknown_strategy, False, "Unknown mixture_priors strategy"),

        # Explicit dict validation errors - sum
        ("PriorsSumOver", _priors_sum_over_one, False, "must sum to 1.0"),
        ("PriorsSumUnder", _priors_sum_under_one, False, "must sum to 1.0"),

        # Explicit dict validation errors - range
        ("PriorNegative", _prior_value_negative, False, "must be in range"),
        ("PriorOverOne", _prior_value_over_one, False, "must be in range"),

        # Explicit dict validation errors - label mismatch
        ("MissingLabelInPriors", _missing_label_in_priors, False, "Labels missing in mixture_priors"),
        ("ExtraLabelInPriors", _extra_label_in_priors, False, "Extra keys in mixture_priors"),

        # Type validation errors
        # Note: These fail at Pydantic's type validation level (before custom validator runs)
        ("NonNumericPrior", _non_numeric_prior, False, "Input should be a valid"),
        ("PriorsWrongType", _priors_wrong_type, False, "Input should be a valid"),
        ("EmptyLabelsList", _empty_labels_list, False, "List should have at least 1 item"),
    ]
)
def test_inference_experiment_parametrized(_test_name, modify_exp, should_pass, error_match, mock_valid_inference_experiment):
    """Test InferenceExperiment validation with various valid and invalid inputs."""
    exp_copy = copy.deepcopy(mock_valid_inference_experiment)
    modified_exp = modify_exp(exp_copy)

    if should_pass:
        exp = InferenceExperiment(**modified_exp)
        # Verify mixture_priors is now a dict (even if it was a strategy string)
        assert isinstance(exp.mixture_priors, dict)
        # Verify sum is 1.0
        assert abs(sum(exp.mixture_priors.values()) - 1.0) < 1e-10
    else:
        with pytest.raises(ValidationError, match=error_match):
            InferenceExperiment(**modified_exp)


def test_inference_equal_split_expansion():
    """Test that 'equal_split' strategy correctly expands to dict with equal weights."""
    exp = InferenceExperiment(
        possible_composition_labels=['Dy100', 'Ho100', 'Tb100'],
        mixture_priors='equal_split',
        beam_current=BeamCurrent(estimate=0.18, unit='nA', type='nominal'),
        raw_config='default_raw_config',
        processed_config='default_processed_config',
        metadata_templates='default_metadata_templates'
    )

    # Should expand to exactly 1/3 for each
    assert exp.mixture_priors == {'Dy100': 1/3, 'Ho100': 1/3, 'Tb100': 1/3}
    # Should sum exactly to 1.0
    assert abs(sum(exp.mixture_priors.values()) - 1.0) < 1e-10


def test_inference_equal_split_six_classes():
    """Test 'equal_split' with 6 classes (realistic inference case)."""
    labels = ['Dy100', 'Ho100', 'Dy80Ho20', 'Dy65Ho35', 'Dy50Ho50', 'Dy30Ho70']

    exp = InferenceExperiment(
        possible_composition_labels=labels,
        mixture_priors='equal_split',
        beam_current=BeamCurrent(estimate=0.18, unit='nA', type='nominal'),
        raw_config='default_raw_config',
        processed_config='default_processed_config',
        metadata_templates='default_metadata_templates'
    )

    # Each should be exactly 1/6
    for label in labels:
        assert exp.mixture_priors[label] == 1/6

    # Should sum exactly to 1.0
    assert sum(exp.mixture_priors.values()) == 1.0


# ===================================================================
# ExperimentDataset Tests
# ===================================================================

def test_experiment_dataset_training_valid():
    """Test ExperimentDataset with valid training data."""
    training_data = {
        'physical_single_particle': {
            '2024-06-29_Dy100': {
                'composition_label': 'Dy100',
                'beam_current': {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'},
                'raw_config': 'default_raw_config',
                'processed_config': 'default_processed_config',
                'metadata_templates': 'default_metadata_templates'
            }
        }
    }

    dataset = ExperimentDataset(dataset_type='training_data', **training_data)

    # Verify group names
    assert 'physical_single_particle' in dataset.get_group_names()

    # Verify experiment IDs
    assert '2024-06-29_Dy100' in dataset.get_experiment_ids('physical_single_particle')

    # Verify we can get the experiment
    exp = dataset.get_group_experiments('physical_single_particle')['2024-06-29_Dy100']
    assert isinstance(exp, TrainingExperiment)
    assert exp.composition_label == 'Dy100'


def test_experiment_dataset_inference_valid():
    """Test ExperimentDataset with valid inference data."""
    inference_data = {
        'physical_multi_particle': {
            '2025-07-02_mixture': {
                'possible_composition_labels': ['Dy100', 'Ho100'],
                'mixture_priors': 'equal_split',
                'beam_current': {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'},
                'raw_config': 'default_raw_config',
                'processed_config': 'default_processed_config',
                'metadata_templates': 'default_metadata_templates'
            }
        }
    }

    dataset = ExperimentDataset(dataset_type='inference_data', **inference_data)

    # Verify experiment type is InferenceExperiment
    exp = dataset.get_group_experiments('physical_multi_particle')['2025-07-02_mixture']
    assert isinstance(exp, InferenceExperiment)
    assert exp.mixture_priors == {'Dy100': 0.5, 'Ho100': 0.5}


def test_experiment_dataset_invalid_dataset_type():
    """Test ExperimentDataset with invalid dataset_type."""
    with pytest.raises(ValueError, match="dataset_type must be 'training_data' or 'inference_data'"):
        ExperimentDataset(dataset_type='unknown_data', physical_single_particle={})


def test_experiment_dataset_group_not_dict():
    """Test ExperimentDataset when group is not a dict."""
    with pytest.raises(ValueError, match="Group 'bad_group' must be a dict"):
        ExperimentDataset(dataset_type='training_data', bad_group="not_a_dict")


def test_experiment_dataset_experiment_not_dict():
    """Test ExperimentDataset when experiment config is not a dict."""
    with pytest.raises(ValueError, match="Experiment 'exp1' in group 'group1' must be a dict"):
        ExperimentDataset(
            dataset_type='training_data',
            group1={'exp1': "not_a_dict"}
        )


def test_experiment_dataset_wrong_experiment_type_in_training():
    """Test that inference experiment in training dataset raises error."""
    inference_exp = {
        'possible_composition_labels': ['Dy100', 'Ho100'],
        'mixture_priors': 'equal_split',
        'beam_current': {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'},
        'raw_config': 'default_raw_config',
        'processed_config': 'default_processed_config',
        'metadata_templates': 'default_metadata_templates'
    }

    # TrainingExperiment requires composition_label, not possible_composition_labels
    with pytest.raises(ValidationError, match="Field required"):
        ExperimentDataset(
            dataset_type='training_data',
            physical_single_particle={'exp1': inference_exp}
        )


def test_experiment_dataset_wrong_experiment_type_in_inference():
    """Test that training experiment in inference dataset raises error."""
    training_exp = {
        'composition_label': 'Dy100',
        'beam_current': {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'},
        'raw_config': 'default_raw_config',
        'processed_config': 'default_processed_config',
        'metadata_templates': 'default_metadata_templates'
    }

    # InferenceExperiment requires possible_composition_labels, not composition_label
    with pytest.raises(ValidationError, match="Field required"):
        ExperimentDataset(
            dataset_type='inference_data',
            physical_multi_particle={'exp1': training_exp}
        )


# ===================================================================
# ExperimentsDatabase Tests
# ===================================================================

@pytest.fixture
def mock_valid_experiments_database() -> dict:
    """Provides a fixture for a valid ExperimentsDatabase."""
    return {
        'project_constants': {
            'substrate': 'Si wafer',
            'host_matrix': 'NaF4',
            'instrument_model': 'Zeiss SUPRA',
            'CL_detection_system': {
                'mirror': {'type': 'parabolic'},
                'channels': {'blue': {}, 'green': {}, 'red': {}}
            },
            'SE_detection_system': {'detector_type': 'Everhart-Thornley'}
        },
        'training_data': {
            'physical_single_particle': {
                '2024-06-29_Dy100': {
                    'composition_label': 'Dy100',
                    'beam_current': {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'},
                    'raw_config': 'default_raw_config',
                    'processed_config': 'default_processed_config',
                    'metadata_templates': 'default_metadata_templates'
                }
            }
        },
        'inference_data': {
            'physical_multi_particle': {
                '2025-07-02_mixture': {
                    'possible_composition_labels': ['Dy100', 'Ho100'],
                    'mixture_priors': 'equal_split',
                    'beam_current': {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'},
                    'raw_config': 'default_raw_config',
                    'processed_config': 'default_processed_config',
                    'metadata_templates': 'default_metadata_templates'
                }
            }
        }
    }


def test_experiments_database_valid(mock_valid_experiments_database):
    """Test ExperimentsDatabase with valid complete structure."""
    db = ExperimentsDatabase(**mock_valid_experiments_database)

    # Verify project constants
    assert db.project_constants.substrate == 'Si wafer'
    assert db.project_constants.instrument_model == 'Zeiss SUPRA'

    # Verify training dataset
    training = db.get_training_dataset()
    assert isinstance(training, ExperimentDataset)
    assert 'physical_single_particle' in training.get_group_names()

    # Verify inference dataset
    inference = db.get_inference_dataset()
    assert isinstance(inference, ExperimentDataset)
    assert 'physical_multi_particle' in inference.get_group_names()


def test_experiments_database_get_all_experiments(mock_valid_experiments_database):
    """Test get_all_experiments() returns correct structure."""
    db = ExperimentsDatabase(**mock_valid_experiments_database)

    all_exps = db.get_all_experiments()

    # Verify structure
    assert 'training_data' in all_exps
    assert 'inference_data' in all_exps

    # Verify training experiments
    assert 'physical_single_particle' in all_exps['training_data']
    assert '2024-06-29_Dy100' in all_exps['training_data']['physical_single_particle']

    # Verify inference experiments
    assert 'physical_multi_particle' in all_exps['inference_data']
    assert '2025-07-02_mixture' in all_exps['inference_data']['physical_multi_particle']


def test_experiments_database_missing_project_constants():
    """Test ExperimentsDatabase with missing project_constants."""
    with pytest.raises(ValidationError, match="Field required"):
        ExperimentsDatabase(
            training_data={},
            inference_data={}
        )


def test_experiments_database_missing_training_data(mock_valid_experiments_database):
    """Test ExperimentsDatabase with missing training_data."""
    data = copy.deepcopy(mock_valid_experiments_database)
    data.pop('training_data')

    with pytest.raises(ValidationError, match="Field required"):
        ExperimentsDatabase(**data)


def test_experiments_database_missing_inference_data(mock_valid_experiments_database):
    """Test ExperimentsDatabase with missing inference_data."""
    data = copy.deepcopy(mock_valid_experiments_database)
    data.pop('inference_data')

    with pytest.raises(ValidationError, match="Field required"):
        ExperimentsDatabase(**data)


def test_experiments_database_invalid_nested_experiment(mock_valid_experiments_database):
    """Test that invalid experiment in nested structure raises error (propagation test)."""
    data = copy.deepcopy(mock_valid_experiments_database)
    # Set invalid beam_current in nested experiment
    data['training_data']['physical_single_particle']['2024-06-29_Dy100']['beam_current']['estimate'] = -0.1

    with pytest.raises(ValidationError, match="greater than 0"):
        ExperimentsDatabase(**data)
