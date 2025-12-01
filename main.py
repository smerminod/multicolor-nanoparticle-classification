"""
Main entry point for the nanoparticle analysis pipeline.

This script serves as a demonstration and smoke test for the project's
configuration system. It validates all YAML configurations and demonstrates
the config_loader module's functionality by loading, merging, and displaying
experiment configurations.

Usage:
    python main.py [experiment_id_1] [experiment_id_2] ...

Examples:
    # Show all available experiments
    python main.py

    # Demonstrate configuration loading for specific experiment(s)
    python main.py 2024-06-29_Dy100
    python main.py 2024-06-29_Dy100 2025-07-02_6-component-mixture
"""

import glob
import sys
from pathlib import Path

from src.data import config_loader


def demonstrate_config_loading(exp_id: str, validated_configs=None) -> None:
    """
    Loads all configurations for a given experiment ID and prints a summary.
    This function demonstrates the core functionality of the config_loader module.

    Args:
        exp_id: Experiment ID to process (e.g., '2024-06-29_Dy100')
        validated_configs: Optional pre-validated configs from verify_experiment_references()
    """
    print(f"\n{'='*25} Demonstrating for Experiment: {exp_id} {'='*25}")

    try:
        # 1. Get the full, merged configuration for the experiment
        print("\n1. Loading full, merged experiment configuration...")
        full_config, dataset, group = config_loader.get_full_experiment_config(
            exp_id, validated_configs
        )
        print(f"   ✅ Success! Found in: {dataset} -> {group}")

        # Print key details from the configuration (different for training vs inference)
        if 'composition_label' in full_config:  # Training data
            print(f"      - Known composition: {full_config['composition_label']}")
            print(f"      - Elemental fractions: {full_config['elemental_fractions']}")
        elif 'possible_composition_labels' in full_config:  # Inference data
            labels = full_config['possible_composition_labels']
            print(f"      - Candidate compositions: {len(labels)} classes ({', '.join(labels)})")
            print(f"      - Mixture priors: {full_config['mixture_priors']}")

        # Other common details
        print(f"      - Host matrix: {full_config.get('host_matrix')}")
        print(f"      - Beam current: {full_config.get('beam_current', {}).get('estimate')} {full_config.get('beam_current', {}).get('unit')}")
        print(f"      - Available processing methods: {list(full_config.get('processed_config', {}).keys())}")

        # 2. Get all the essential directory paths
        print("\n2. Constructing experiment directory paths...")
        paths = config_loader.get_experiment_paths(exp_id, validated_configs)
        print(f"   ✅ Success! Base path: {paths['base']}")
        print(f"      - Raw data: {paths['raw']}")
        print(f"      - Processed data: {paths['processed']}")
        print(f"      - Metadata: {paths['metadata']}")

        # 3. Get the path to a specific metadata file
        print("\n3. Constructing metadata file path for 'regions' metadata type...")
        regions_csv_path = config_loader.get_metadata_filepath(
            exp_id, metadata_type='regions', validated_configs=validated_configs
        )
        print(f"   ✅ Success! Regions CSV path: {regions_csv_path}")

        # 4. Assemble the raw file path patterns for a sample region
        print("\n4. Assembling raw file path patterns for 'region_1'...")
        raw_filepaths = config_loader.get_raw_filepaths(
            exp_id, region_id='region_1', validated_configs=validated_configs
        )
        if raw_filepaths:
            print(f"   ✅ Success! Found {len(raw_filepaths)} raw data patterns:")
            for key, path in raw_filepaths.items():
                print(f"      - {key}: {Path(path).name}")
        else:
            print("   ⚠️  Failed to get raw file paths (check configuration)")

        # 5. Assemble the processed file path patterns for a sample region and method
        print("\n5. Assembling processed file path patterns for 'summed_images' method in 'region_1'...")
        processed_filepaths = config_loader.get_processed_filepaths(
            exp_id,
            method_name='summed_images',
            region_id='region_1',
            validated_configs=validated_configs
        )
        if processed_filepaths:
            print(f"   ✅ Success! Found {len(processed_filepaths)} processed output patterns:")
            for key, path in list(processed_filepaths.items())[:3]:  # Show first 3 only
                print(f"      - {key}: {Path(path).name}")
        else:
            print("   ⚠️  Failed to get processed file paths (check configuration)")

        # 6. Check if sample data exists before trying to use the patterns
        print("\n6. Checking for sample data files...")
        if raw_filepaths and 'CL_red' in raw_filepaths:
            CL_red_pattern = raw_filepaths['CL_red']
            matching_files = glob.glob(str(CL_red_pattern))
            if matching_files:
                print(f"   ✅ Found sample raw data: {Path(matching_files[0]).name}")
            else:
                print("   ⚠️  Sample data not yet included in repository.")
                print(f"     - Raw data for 'CL_red' would go here: {CL_red_pattern}")
                if processed_filepaths:
                    print(f"     - Processed data for 'CL_red' would go here: {processed_filepaths.get('CL_red', 'N/A')}")

    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"  ❌ ERROR: Could not process experiment '{exp_id}'.")
        print(f"     Reason: {e}")


def validate_all_configs():
    """
    Validates all configuration files using the comprehensive validation system.

    Returns:
        Tuple of (success: bool, validated_configs or None)
    """
    print("\n" + "="*80)
    print("  VALIDATING CONFIGURATION FILES")
    print("="*80)

    try:
        # Comprehensive validation silently, then provide our own summary
        validated_configs = config_loader.verify_experiment_references(verbose=False)
        class_defs, pipeline_config, _ = validated_configs

        print("\n✅ All configuration files validated successfully!")
        print(f"   - Class definitions: {len(class_defs.get_class_labels())} classes")
        print(f"   - Pipeline config: {len(pipeline_config.definitions)} template(s)")
        print("   - Cross-references verified")

        return True, validated_configs

    except (ValueError, FileNotFoundError) as e:
        print("\n❌ CRITICAL ERROR: Configuration validation failed.")
        print(f"   Reason: {e}")
        print("\nPlease fix the configuration errors before proceeding.")
        return False, None


def show_available_experiments() -> None:
    """Displays all available experiments organized by dataset type and group."""
    print("\n" + "="*80)
    print("  AVAILABLE EXPERIMENTS")
    print("="*80)
    print(config_loader.format_experiments_summary())


def main() -> None:
    """
    Main entry point. Validates configurations and demonstrates the config system
    for specified experiments.
    """
    print("\n" + "="*80)
    print("  **  NANOPARTICLE CLASSIFICATION PIPELINE - Configuration Demo **")
    print("="*80)

    # Step 1: Validate all configuration files once then reuse
    success, validated_configs = validate_all_configs()
    if not success:
        sys.exit(1)  # Exit with error code if validation fails

    # Step 2: Parse command-line arguments
    experiment_ids_to_process = sys.argv[1:]

    if not experiment_ids_to_process:
        # No experiments specified - show all available experiments
        show_available_experiments()
        print("\n" + "="*80)
        print("Usage: python main.py <experiment_id_1> <experiment_id_2> ...")
        print("="*80)
        return

    # Step 3: Demonstrate configuration loading for each specified experiment
    # IMPORTANT: Reuse validated_configs to avoid redundant validation
    print("\n" + "="*80)
    print(f"  DEMONSTRATING CONFIGURATION SYSTEM FOR {len(experiment_ids_to_process)} EXPERIMENT(S)")
    print("  (Reusing validated configs for optimal performance)")
    print("="*80)

    for exp_id in experiment_ids_to_process:
        demonstrate_config_loading(exp_id, validated_configs)

    print("\n" + "="*80)
    print("  Demo completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
