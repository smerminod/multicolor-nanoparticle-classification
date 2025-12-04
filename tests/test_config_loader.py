"""
Test suite for the src.data.config_loader module.

This suite uses pytest and mocking to create isolated unit tests for the core
configuration loading and path generation logic, ensuring the system is robust
and maintainable.
"""

import copy
from pathlib import Path

import pytest

from src.data import config_loader

# Note: In parametrized tests, the first parameter (_test_name) is used by pytest
# to generate test IDs but is not accessed in the test body. This is intentional.

# Constants for testing
VALID_EXP_ID = '2024-06-29_Dy100'
INVALID_EXP_ID = 'nonexistent-id'


# ===================================================================
# Pytest Fixtures
# ===================================================================

@pytest.fixture
def mock_class_definitions() -> dict:
    """Provides a fixture for a valid class_definitions.yaml file."""
    return {
        'nanoparticle_classes': {
            'Dy100': {
                'elemental_fractions': {'Tb': 0.0, 'Dy': 1.0, 'Ho': 0.0, 'Gd': 0.0, 'Y': 0.0}
            },
            'Ho100': {
                'elemental_fractions': {'Tb': 0.0, 'Dy': 0.0, 'Ho': 1.0, 'Gd': 0.0, 'Y': 0.0}
            }
        }
    }

@pytest.fixture
def mock_pipeline_config() -> dict:
    """Provides fixture for a valid config.yaml file."""
    return {
        'definitions': {
            'default_raw_config': {
                'acquisitions': {
                    'CL_SE': {
                        'CL': {
                            'blue': {'source_prefix': '593CP_NP_', 'target_dopant': 'Tb'},
                            'green': {'source_prefix': '593SP_NP_', 'target_dopant': 'Dy'},
                            'red': {'source_prefix': '593LP_NP_', 'target_dopant': 'Ho'}
                        },
                        'SE': {'source_prefix': 'SE2_NP_', 'target_dopant': None}
                    },
                    'SEM': {'source_prefix': 'NP_', 'target_dopant': None}
                }
            },
            'default_processed_config': {
                'summed_images': {
                    'outputs': {
                        'CL_SE': {
                            'CL': {
                                'blue': {'filename_prefix': 'summed_blue'},
                                'green': {'filename_prefix': 'summed_green'},
                                'red': {'filename_prefix': 'summed_red'}
                            },
                            'SE': {'filename_prefix': 'summed_SE'}
                        },
                        'SEM': {'filename_prefix': 'summed_SEM'}
                    }
                }
            },
            'default_metadata_templates': {
                'regions': '{id}_regions.csv'
            }
        }
    }

@pytest.fixture
def mock_experiments_db() -> dict:
    """Provides fixture for a valid experiments.yaml file."""
    return {
        'project_constants': {
            'substrate': 'Test substrate',
            'host_matrix': 'Test Matrix',
            'instrument_model': 'Test SEM',
            'CL_detection_system': {
                'mirror': {'type': 'parabolic'},
                'channels': {'blue': {}, 'green': {}, 'red': {}}
            },
            'SE_detection_system': {'detector_type': 'Everhart-Thornley'}
        },
        'training_data': {
            'physical_single_particle': {
                VALID_EXP_ID: {
                    'composition_label': 'Dy100',
                    'beam_current': {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'},
                    'raw_config': 'default_raw_config',
                    'processed_config': 'default_processed_config',
                    'metadata_templates': 'default_metadata_templates'
                }
            },
            'simulation_single_particle': {
                '2025-12-01_Ho100_sim': {
                    'composition_label': 'Ho100',
                    'beam_current': {'estimate': 0.20, 'unit': 'nA', 'type': 'nominal'},
                    'raw_config': 'default_raw_config',
                    'processed_config': 'default_processed_config',
                    'metadata_templates': 'default_metadata_templates'
                }
            }
        },
        'inference_data': {
            'physical_multi_particle': {
                '2025-12-02_6-component-mixture': {
                    'possible_composition_labels': ['Dy100', 'Ho100'],
                    'mixture_priors': {'Dy100': 0.3, 'Ho100': 0.7},
                    'beam_current': {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'},
                    'raw_config': 'default_raw_config',
                    'processed_config': 'default_processed_config',
                    'metadata_templates': 'default_metadata_templates'
                }
            }
        }
    }


# ===================================================================
# Unit Tests for verify_class_definitions
# ===================================================================

def test_verify_class_definitions_calls_schema(mocker, mock_class_definitions):
    """Test that verify_class_definitions correctly validates using ClassDefinitions schema."""
    # ARRANGE
    mocker.patch('src.data.config_loader.load_class_definitions', return_value=mock_class_definitions)

    # ACT
    result = config_loader.verify_class_definitions()

    # ASSERT
    assert result is not None
    assert 'Dy100' in result.get_class_labels()
    assert 'Ho100' in result.get_class_labels()

def test_verify_class_definitions_propagates_validation_error(mocker):
    """Test that schema validation errors are re-raised as ValueError."""
    # ARRANGE: Simulate missing nanoparticle classes
    invalid_data = {'nanoparticle_classes': {}}
    mocker.patch('src.data.config_loader.load_class_definitions', return_value=invalid_data)

    # ACT & ASSERT
    with pytest.raises(ValueError, match="Invalid class definitions"):
        config_loader.verify_class_definitions()


# ===================================================================
# Unit Tests for verify_pipeline_config
# ===================================================================

def test_verify_pipeline_config_calls_schema(mocker, mock_pipeline_config):
    """Test that verify_pipeline_config correctly validates using PipelineConfig schema."""
    # ARRANGE
    mocker.patch('src.data.config_loader.load_pipeline_config', return_value=mock_pipeline_config)

    # ACT
    result = config_loader.verify_pipeline_config()

    # ASSERT
    assert result is not None
    assert 'default_raw_config' in result.definitions
    assert 'default_processed_config' in result.definitions
    assert 'default_metadata_templates' in result.definitions

def test_verify_pipeline_config_propagates_validation_error(mocker):
    """Test that schema validation errors are re-raised as ValueError."""
    # ARRANGE: Simulate missing required templates
    invalid_data = {'definitions': {}}
    mocker.patch('src.data.config_loader.load_pipeline_config', return_value=invalid_data)

    # ACT & ASSERT
    with pytest.raises(ValueError, match="Invalid pipeline configuration"):
        config_loader.verify_pipeline_config()


# ===================================================================
# Unit Tests for verify_experiment_config
# ===================================================================

def test_verify_experiment_config_calls_schema(mocker, mock_experiments_db):
    """Test that verify_experiment_config correctly validates using ExperimentsDatabase schema."""
    # ARRANGE
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=mock_experiments_db)

    # ACT
    result = config_loader.verify_experiment_config()

    # ASSERT
    assert result is not None
    training_ds = result.get_training_dataset()
    assert 'physical_single_particle' in training_ds.get_group_names()

def test_verify_experiment_config_propagates_validation_error(mocker):
    """Test that schema validation errors are re-raised as ValueError."""
    # ARRANGE: Simulate invalid structure (missing project_constants)
    invalid_data = {'training_data': {}, 'inference_data': {}}
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=invalid_data)

    # ACT & ASSERT
    with pytest.raises(ValueError, match="Invalid experiments database"):
        config_loader.verify_experiment_config()


# ===================================================================
# Unit Tests for verify_experiment_references (cross-validation)
# ===================================================================

def test_verify_experiment_references_success(mocker, mock_class_definitions, mock_pipeline_config, mock_experiments_db):
    """Test that verify_experiment_references succeeds with valid cross-references."""
    # ARRANGE
    mocker.patch('src.data.config_loader.load_class_definitions', return_value=mock_class_definitions)
    mocker.patch('src.data.config_loader.load_pipeline_config', return_value=mock_pipeline_config)
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=mock_experiments_db)

    # ACT
    class_defs, pipeline_config, experiments_db = config_loader.verify_experiment_references()

    # ASSERT
    assert class_defs is not None
    assert pipeline_config is not None
    assert experiments_db is not None
    assert 'Dy100' in class_defs.get_class_labels()

def test_verify_experiment_references_missing_composition_label(mocker, mock_class_definitions, mock_pipeline_config, mock_experiments_db):
    """Test that missing composition_label in class_definitions raises ValueError."""
    # ARRANGE: Remove Ho100 from class definitions
    class_defs_copy = copy.deepcopy(mock_class_definitions)
    class_defs_copy['nanoparticle_classes'].pop('Ho100')

    mocker.patch('src.data.config_loader.load_class_definitions', return_value=class_defs_copy)
    mocker.patch('src.data.config_loader.load_pipeline_config', return_value=mock_pipeline_config)
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=mock_experiments_db)

    # ACT & ASSERT
    with pytest.raises(ValueError, match="references composition_label 'Ho100' which doesn't exist"):
        config_loader.verify_experiment_references()

def test_verify_experiment_references_missing_possible_label(mocker, mock_class_definitions, mock_pipeline_config, mock_experiments_db):
    """Test that missing label in possible_composition_labels raises ValueError."""
    # ARRANGE: Create an inference experiment with a class that doesn't exist
    experiments_copy = copy.deepcopy(mock_experiments_db)
    # Add 'He100' to inference experiment's possible labels and priors (it doesn't exist in class_definitions)
    inf_exp = experiments_copy['inference_data']['physical_multi_particle']['2025-12-02_6-component-mixture']
    inf_exp['possible_composition_labels'] = ['Dy100', 'Ho100', 'He100']
    inf_exp['mixture_priors'] = {'Dy100': 0.2, 'Ho100': 0.5, 'He100': 0.3}  # Matching prior

    mocker.patch('src.data.config_loader.load_class_definitions', return_value=mock_class_definitions)
    mocker.patch('src.data.config_loader.load_pipeline_config', return_value=mock_pipeline_config)
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=experiments_copy)

    # ACT & ASSERT: Should fail on inference experiment with missing He100
    with pytest.raises(ValueError, match="references possible_composition_label 'He100' which doesn't exist"):
        config_loader.verify_experiment_references()

def test_verify_experiment_references_missing_template(mocker, mock_class_definitions, mock_pipeline_config, mock_experiments_db):
    """Test that missing config template reference raises ValueError during cross-validation."""
    # ARRANGE: To an experiment, add a template reference that doesn't exist in pipeline_config
    experiments_copy = copy.deepcopy(mock_experiments_db)
    experiments_copy['training_data']['physical_single_particle'][VALID_EXP_ID]['raw_config'] = 'custom_raw_config'

    mocker.patch('src.data.config_loader.load_class_definitions', return_value=mock_class_definitions)
    mocker.patch('src.data.config_loader.load_pipeline_config', return_value=mock_pipeline_config)
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=experiments_copy)

    # ACT & ASSERT: Cross-validation should catch the missing template reference
    with pytest.raises(ValueError, match="references raw_config 'custom_raw_config' which doesn't exist"):
        config_loader.verify_experiment_references()


# ===================================================================
# Unit Tests for get_all_experiments
# ===================================================================

def test_get_all_experiments(mocker, mock_experiments_db):
    """
    Tests that get_all_experiments correctly scans and organizes all
    experiment IDs from the experiments database.
    """
    # ARRANGE
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=mock_experiments_db)

    # ACT
    all_experiments = config_loader.get_all_experiments()

    # ASSERT: Check structure and contents
    assert isinstance(all_experiments, dict)
    assert VALID_EXP_ID in all_experiments['training_data']['physical_single_particle']
    assert '2025-12-01_Ho100_sim' in all_experiments['training_data']['simulation_single_particle']
    assert '2025-12-02_6-component-mixture' in all_experiments['inference_data']['physical_multi_particle']

def test_get_all_experiments_empty_groups(mocker):
    """Tests that get_all_experiments handles empty experiment groups gracefully."""
    # ARRANGE: Create a database with some empty groups
    mock_db_edge_case = {
        'project_constants': {
            'substrate': 'Test',
            'host_matrix': 'Test',
            'instrument_model': 'Test',
            'CL_detection_system': {'mirror': {}},
            'SE_detection_system': {'detector': 'Test'}
        },
        'training_data': {
            'physical_single_particle': {
                '2025-12-01_Dy100': {
                    'composition_label': 'Dy100',
                    'beam_current': {'estimate': 0.18, 'unit': 'nA', 'type': 'nominal'},
                    'raw_config': 'default_raw_config',
                    'processed_config': 'default_processed_config',
                    'metadata_templates': 'default_metadata_templates'
                }
            },
            'empty_group': {}
        },
        'inference_data': {}
    }
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=mock_db_edge_case)

    # ACT: Run the function
    all_experiments = config_loader.get_all_experiments()

    # ASSERT: Empty groups should appear in result but be empty lists
    assert 'empty_group' in all_experiments['training_data']
    assert all_experiments['training_data']['empty_group'] == []
    assert 'physical_single_particle' in all_experiments['training_data']
    assert len(all_experiments['training_data']['physical_single_particle']) == 1


# ===================================================================
# Unit Tests for format_experiments_summary
# ===================================================================

def test_format_experiments_summary(mocker, mock_class_definitions, mock_pipeline_config, mock_experiments_db):
    """Test format_experiments_summary outputs expected content."""
    # ARRANGE
    mocker.patch('src.data.config_loader.load_class_definitions', return_value=mock_class_definitions)
    mocker.patch('src.data.config_loader.load_pipeline_config', return_value=mock_pipeline_config)
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=mock_experiments_db)

    # ACT
    summary = config_loader.format_experiments_summary(include_stats=True)

    # ASSERT
    assert isinstance(summary, str)
    assert VALID_EXP_ID in summary
    assert "2025-12-02_6-component-mixture" in summary
    assert "2 training experiment(s), 1 inference experiment(s)" in summary


# ===================================================================
# Unit Tests for get_full_experiment_config
# ===================================================================

def test_get_full_experiment_config_training_success(mocker, mock_experiments_db, mock_pipeline_config, mock_class_definitions):
    """
    Tests that get_full_experiment_config successfully finds and merges data
    from all three mocked source files for a training experiment.
    """
    # ARRANGE: Mock the file loaders with the data provided by the fixtures
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=mock_experiments_db)
    mocker.patch('src.data.config_loader.load_pipeline_config', return_value=mock_pipeline_config)
    mocker.patch('src.data.config_loader.load_class_definitions', return_value=mock_class_definitions)

    # ACT: Run the function we are testing
    full_config, dataset_type, group_name = config_loader.get_full_experiment_config(VALID_EXP_ID)

    # ASSERT: Check that the output is correctly merged and structured
    assert isinstance(full_config, dict)
    assert dataset_type == 'training_data'
    assert group_name == 'physical_single_particle'

    # Check that data from experiments.yaml was merged successfully
    assert full_config.get('composition_label') == 'Dy100'
    assert full_config.get('host_matrix') == 'Test Matrix'

    # Check that data from class_definitions.yaml and config.yaml were merged successfully
    assert isinstance(full_config.get('elemental_fractions'), dict)
    assert isinstance(full_config.get('raw_config'), dict)
    assert isinstance(full_config.get('processed_config'), dict)
    assert isinstance(full_config.get('metadata_templates'), dict)

    # Check a specific value from each merged dictionary to confirm the merge was correct
    assert full_config['elemental_fractions']['Dy'] == 1.0
    assert full_config['raw_config']['acquisitions']['SEM']['source_prefix'] == 'NP_'
    assert full_config['processed_config']['summed_images']['outputs']['CL_SE']['SE']['filename_prefix'] == 'summed_SE'
    assert full_config['metadata_templates']['regions'] == '{id}_regions.csv'

def test_get_full_experiment_config_inference_success(mocker, mock_experiments_db, mock_pipeline_config, mock_class_definitions):
    """
    Tests that get_full_experiment_config correctly handles inference experiments
    with class_definitions dict and mixture_priors.
    """
    # ARRANGE
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=mock_experiments_db)
    mocker.patch('src.data.config_loader.load_pipeline_config', return_value=mock_pipeline_config)
    mocker.patch('src.data.config_loader.load_class_definitions', return_value=mock_class_definitions)

    # ACT
    full_config, dataset_type, group_name = config_loader.get_full_experiment_config('2025-12-02_6-component-mixture')

    # ASSERT
    assert dataset_type == 'inference_data'
    assert group_name == 'physical_multi_particle'
    assert 'possible_composition_labels' in full_config
    assert 'mixture_priors' in full_config
    assert 'class_definitions' in full_config
    assert 'Dy100' in full_config['class_definitions']
    assert 'Ho100' in full_config['class_definitions']
    assert full_config['mixture_priors']['Dy100'] == 0.3
    assert full_config['mixture_priors']['Ho100'] == 0.7

def test_get_full_experiment_config_failure(mocker, mock_experiments_db, mock_pipeline_config, mock_class_definitions):
    """Tests that get_full_experiment_config raises a ValueError for a nonexistent ID."""
    # ARRANGE
    mocker.patch('src.data.config_loader.load_experiments_db', return_value=mock_experiments_db)
    mocker.patch('src.data.config_loader.load_pipeline_config', return_value=mock_pipeline_config)
    mocker.patch('src.data.config_loader.load_class_definitions', return_value=mock_class_definitions)

    # ACT & ASSERT
    with pytest.raises(ValueError, match="not found in experiments.yaml"):
        config_loader.get_full_experiment_config(INVALID_EXP_ID)


# ===================================================================
# Unit Tests for Path Generation Functions
# ===================================================================

def test_get_experiment_paths(mocker):
    """Tests that get_experiment_paths correctly constructs directory paths."""
    # ARRANGE
    mock_return = ({}, 'training_data', 'physical_single_particle')
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=mock_return)

    # ACT
    paths = config_loader.get_experiment_paths(VALID_EXP_ID)

    # ASSERT
    assert paths['base'].name == VALID_EXP_ID
    assert 'training' in str(paths['base'])
    assert 'physical' in str(paths['base'])
    assert 'single_particle' in str(paths['base'])
    assert paths['raw'] == paths['base'] / 'raw'
    assert paths['processed'] == paths['base'] / 'processed'
    assert paths['metadata'] == paths['base'] / 'metadata'

def test_get_metadata_filepath(mocker):
    """Tests metadata file path construction with template substitution."""
    # ARRANGE
    mock_full_config = {'metadata_templates': {'regions': '{id}_regions.csv'}}
    mock_paths = {'metadata': Path('/fake/project/data/training/physical/single_particle/exp_id/metadata')}
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_full_config, '', ''))
    mocker.patch('src.data.config_loader.get_experiment_paths', return_value=mock_paths)

    # ACT
    path = config_loader.get_metadata_filepath(VALID_EXP_ID, metadata_type='regions')

    # ASSERT
    assert path.name == f"{VALID_EXP_ID}_regions.csv"
    assert str(path) == f"/fake/project/data/training/physical/single_particle/exp_id/metadata/{VALID_EXP_ID}_regions.csv"

def test_get_raw_filepaths_success(mocker, mock_pipeline_config):
    """Tests that raw filepath patterns are constructed correctly by walking the config."""
    # ARRANGE
    mock_paths = {'raw': Path('/fake/project/data/some_dirs/raw')}
    mock_full_config = {'raw_config': mock_pipeline_config['definitions']['default_raw_config']}
    mocker.patch('src.data.config_loader.get_experiment_paths', return_value=mock_paths)
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_full_config, '', ''))

    # ACT
    region_to_test = 'region_1'
    filepaths = config_loader.get_raw_filepaths(VALID_EXP_ID, region_id=region_to_test)

    # ASSERT
    assert isinstance(filepaths, dict)

    # Check that all expected keys are present
    expected_keys = ['SEM', 'SE', 'CL_blue', 'CL_green', 'CL_red']
    for key in expected_keys:
        assert key in filepaths

    # Check that the paths are constructed correctly
    assert str(filepaths['SEM']) == f"/fake/project/data/some_dirs/raw/{region_to_test}/SEM/NP_*"
    assert str(filepaths['SE']) == f"/fake/project/data/some_dirs/raw/{region_to_test}/CL_SE/SE/SE2_NP_*"
    assert str(filepaths['CL_blue']) == f"/fake/project/data/some_dirs/raw/{region_to_test}/CL_SE/CL/blue/593CP_NP_*"
    assert str(filepaths['CL_green']) == f"/fake/project/data/some_dirs/raw/{region_to_test}/CL_SE/CL/green/593SP_NP_*"
    assert str(filepaths['CL_red']) == f"/fake/project/data/some_dirs/raw/{region_to_test}/CL_SE/CL/red/593LP_NP_*"

def test_get_raw_filepaths_missing_config(mocker, capsys):
    """Tests that get_raw_filepaths returns None when config is incomplete (KeyError)."""
    # ARRANGE
    mock_paths = {'raw': Path('/fake/project/data/some_dirs/raw')}
    mock_incomplete_config = {}  # Missing 'raw_config' key - will raise KeyError
    mocker.patch('src.data.config_loader.get_experiment_paths', return_value=mock_paths)
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_incomplete_config, '', ''))

    # ACT
    region_to_test = 'region_1'
    result = config_loader.get_raw_filepaths(VALID_EXP_ID, region_id=region_to_test)

    # ASSERT: Should return None when KeyError occurs
    assert result is None

    # ASSERT: Should print error message to stderr
    captured = capsys.readouterr()
    assert f"ERROR: Could not find raw config for experiment '{VALID_EXP_ID}'" in captured.err

def test_get_processed_filepaths_success(mocker, mock_pipeline_config):
    """Tests that processed filepath patterns are constructed correctly by walking the config."""
    # ARRANGE
    mock_paths = {'processed': Path('/fake/project/data/some_dirs/processed')}
    mock_full_config = {'processed_config': mock_pipeline_config['definitions']['default_processed_config']}
    mocker.patch('src.data.config_loader.get_experiment_paths', return_value=mock_paths)
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_full_config, '', ''))

    # ACT
    method_to_test = 'summed_images'
    region_to_test = 'region_1'
    filepaths = config_loader.get_processed_filepaths(VALID_EXP_ID, method_name=method_to_test, region_id=region_to_test)

    # ASSERT
    assert isinstance(filepaths, dict)

    # Check that all expected keys are present
    expected_keys = ['SEM', 'SE', 'CL_blue', 'CL_green', 'CL_red']
    for key in expected_keys:
        assert key in filepaths

    # Check that the paths are constructed correctly
    assert str(filepaths['SEM']) == f"/fake/project/data/some_dirs/processed/{method_to_test}/{region_to_test}/SEM/summed_SEM_{region_to_test}.tif"
    assert str(filepaths['SE']) == f"/fake/project/data/some_dirs/processed/{method_to_test}/{region_to_test}/CL_SE/SE/summed_SE_{region_to_test}.tif"
    assert str(filepaths['CL_blue']) == f"/fake/project/data/some_dirs/processed/{method_to_test}/{region_to_test}/CL_SE/CL/blue/summed_blue_{region_to_test}.tif"
    assert str(filepaths['CL_green']) == f"/fake/project/data/some_dirs/processed/{method_to_test}/{region_to_test}/CL_SE/CL/green/summed_green_{region_to_test}.tif"
    assert str(filepaths['CL_red']) == f"/fake/project/data/some_dirs/processed/{method_to_test}/{region_to_test}/CL_SE/CL/red/summed_red_{region_to_test}.tif"

def test_get_processed_filepaths_missing_config(mocker, capsys):
    """Tests that get_processed_filepaths returns None when config is incomplete (KeyError)."""
    # ARRANGE
    mock_paths = {'processed': Path('/fake/project/data/some_dirs/processed')}
    mock_incomplete_config = {}  # Missing 'processed_config' key - will raise KeyError
    mocker.patch('src.data.config_loader.get_experiment_paths', return_value=mock_paths)
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_incomplete_config, '', ''))

    # ACT
    method_to_test = 'summed_images'
    region_to_test = 'region_1'
    result = config_loader.get_processed_filepaths(VALID_EXP_ID, method_name=method_to_test, region_id=region_to_test)

    # ASSERT: Should return None when KeyError occurs
    assert result is None

    # ASSERT: Should print error message to stderr
    captured = capsys.readouterr()
    assert f"ERROR: Could not find processed config for method '{method_to_test}'" in captured.err
    assert f"for experiment '{VALID_EXP_ID}'" in captured.err


# ===================================================================
# Unit Tests for get_raw_channel_info and get_processed_channel_info
# ===================================================================

def test_get_raw_channel_info_nested(mocker, mock_pipeline_config):
    """Test navigating to nested CL channels (CL_SE -> CL -> red)."""
    # ARRANGE
    mock_full_config = {'raw_config': mock_pipeline_config['definitions']['default_raw_config']}
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_full_config, '', ''))

    # ACT
    channel_info = config_loader.get_raw_channel_info(VALID_EXP_ID, 'CL_SE', 'CL', 'red')

    # ASSERT
    assert isinstance(channel_info, dict)
    assert channel_info.get('source_prefix') == '593LP_NP_'
    assert channel_info.get('target_dopant') == 'Ho'

def test_get_raw_channel_info_toplevel(mocker, mock_pipeline_config):
    """Test navigating to a top-level acquisition (SEM)."""
    # ARRANGE
    mock_full_config = {'raw_config': mock_pipeline_config['definitions']['default_raw_config']}
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_full_config, '', ''))

    # ACT
    channel_info = config_loader.get_raw_channel_info(VALID_EXP_ID, 'SEM')

    # ASSERT
    assert isinstance(channel_info, dict)
    assert channel_info.get('source_prefix') == 'NP_'

def test_get_raw_channel_info_detector_level(mocker, mock_pipeline_config):
    """Test navigating to a detector level (CL_SE -> SE)."""
    # ARRANGE
    mock_full_config = {'raw_config': mock_pipeline_config['definitions']['default_raw_config']}
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_full_config, '', ''))

    # ACT
    channel_info = config_loader.get_raw_channel_info(VALID_EXP_ID, 'CL_SE', 'SE')

    # ASSERT
    assert isinstance(channel_info, dict)
    assert channel_info.get('source_prefix') == 'SE2_NP_'

def test_get_raw_channel_info_invalid_path(mocker, mock_pipeline_config, capsys):
    """Test that invalid channel paths return None and print error."""
    # ARRANGE
    mock_full_config = {'raw_config': mock_pipeline_config['definitions']['default_raw_config']}
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_full_config, '', ''))

    # ACT
    channel_info = config_loader.get_raw_channel_info(VALID_EXP_ID, 'INVALID_ACQ', None, None)

    # ASSERT: Should return None for invalid paths
    assert channel_info is None

    # ASSERT: Should print error message
    captured = capsys.readouterr()
    assert "ERROR: Could not find raw info" in captured.out

def test_get_processed_channel_info_nested(mocker, mock_pipeline_config):
    """Test navigating to nested CL channels (CL_SE -> CL -> red) for the summed_images method."""
    # ARRANGE
    mock_full_config = {'processed_config': mock_pipeline_config['definitions']['default_processed_config']}
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_full_config, '', ''))

    # ACT
    channel_info = config_loader.get_processed_channel_info(VALID_EXP_ID, 'summed_images', 'CL_SE', 'CL', 'red')

    # ASSERT
    assert isinstance(channel_info, dict)
    assert channel_info.get('filename_prefix') == 'summed_red'

def test_get_processed_channel_info_simple(mocker, mock_pipeline_config):
    """Test navigating to a mid-level acquisition (SE) for the summed_images method."""
    # ARRANGE
    mock_full_config = {'processed_config': mock_pipeline_config['definitions']['default_processed_config']}
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_full_config, '', ''))

    # ACT
    channel_info = config_loader.get_processed_channel_info(VALID_EXP_ID, 'summed_images', 'CL_SE', 'SE')

    # ASSERT
    assert isinstance(channel_info, dict)
    assert channel_info.get('filename_prefix') == 'summed_SE'

def test_get_processed_channel_info_toplevel(mocker, mock_pipeline_config):
    """Test navigating to a top-level acquisition (SEM) for the summed_images method."""
    # ARRANGE
    mock_full_config = {'processed_config': mock_pipeline_config['definitions']['default_processed_config']}
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_full_config, '', ''))

    # ACT
    channel_info = config_loader.get_processed_channel_info(VALID_EXP_ID, 'summed_images', 'SEM')

    # ASSERT
    assert isinstance(channel_info, dict)
    assert channel_info.get('filename_prefix') == 'summed_SEM'

def test_get_processed_channel_info_invalid_path(mocker, mock_pipeline_config, capsys):
    """Test that invalid channel paths return None and print error for processed config."""
    # ARRANGE
    mock_full_config = {'processed_config': mock_pipeline_config['definitions']['default_processed_config']}
    mocker.patch('src.data.config_loader.get_full_experiment_config', return_value=(mock_full_config, '', ''))

    # ACT
    channel_info = config_loader.get_processed_channel_info(VALID_EXP_ID, 'summed_images', 'INVALID_ACQ', None, None)

    # ASSERT: Should return None for invalid paths
    assert channel_info is None

    # ASSERT: Should print error message
    captured = capsys.readouterr()
    assert "ERROR: Could not find processed info" in captured.out


# ===================================================================
# Integration Test
# ===================================================================

@pytest.mark.integration
def test_integration_with_real_files(mocker):
    """
    An integration test that loads the actual YAML files from the project.
    This verifies that the real files are parseable and that the mocked data
    in the unit tests is a reasonable reflection of reality.
    """
    # Un-mock the file loaders for this specific test
    mocker.stopall()

    # Use a real ID from experiments.yaml
    real_exp_id = '2024-06-29_Dy100'
    full_config, _, _ = config_loader.get_full_experiment_config(real_exp_id)

    # Check for the presence of a few key merged items
    assert full_config is not None
    assert 'composition_label' in full_config
    assert 'host_matrix' in full_config
    assert 'acquisitions' in full_config.get('raw_config', {})
    assert 'summed_images' in full_config.get('processed_config', {})
    assert 'regions' in full_config.get('metadata_templates', {})
    