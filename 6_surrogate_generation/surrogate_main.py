"""
MTL-TABlock:  Surrogate Generation Main Entry Point

"""

import os
import sys
import json
import logging
import argparse
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd

# Import utilities
try:
    from . utils. function_replacer import replace_function_call, FunctionReplacer, FunctionLocation
    from .utils.parentheses_balance import find_ending_index
except ImportError:
    # Handle direct script execution
    sys.path.insert(0, os.path.dirname(os.path. abspath(__file__)))
    from function_replacer import replace_function_call, FunctionReplacer, FunctionLocation
    from parentheses_balance import find_ending_index


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SurrogateMainConfig:
    """Configuration for surrogate generation main process."""
    # Paths
    input_folder: str = "server/output"
    output_folder: str = "server/surrogates"
    logs_folder: str = "logs"

    # Processing options
    use_type_aware: bool = True
    preserve_length: bool = True
    skip_inline_scripts: bool = True

    # Logging
    log_level: int = logging.INFO
    log_to_file: bool = True

    # Statistics
    save_statistics: bool = True


@dataclass
class ProcessingStatistics:
    """Statistics for the surrogate generation process."""
    total_websites: int = 0
    processed_websites: int = 0
    failed_websites: int = 0

    total_functions: int = 0
    success:  int = 0
    failed: int = 0
    skipped_inline: int = 0
    skipped_not_found: int = 0

    by_subtype: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_websites": self.total_websites,
            "processed_websites": self.processed_websites,
            "failed_websites": self.failed_websites,
            "total_functions":  self.total_functions,
            "success": self.success,
            "failed":  self.failed,
            "skipped_inline": self.skipped_inline,
            "skipped_not_found": self.skipped_not_found,
            "success_rate": self. success / max(1, self. total_functions - self.skipped_inline - self.skipped_not_found),
            "by_subtype": self.by_subtype,
            "error_count": len(self. errors)
        }


# ============================================================================
# Utility Functions
# ============================================================================

def setup_logging(
    log_folder: str = "logs",
    log_file: str = None,
    level: int = logging. INFO
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_folder:  Folder for log files
        log_file: Specific log file name (optional)
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger("MTL-TABlock-Surrogate")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console.setFormatter(console_format)
    logger.addHandler(console)

    # File handler
    if log_file:
        os.makedirs(log_folder, exist_ok=True)
        log_path = os.path. join(log_folder, log_file)
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def load_request_response_mapping(filename: str) -> Dict[str, str]:
    """
    Load mapping from script URL to request ID.

    Args:
        filename: Path to request. json file

    Returns:
        Dictionary mapping URL to request ID
    """
    try:
        dataset = pd.read_json(filename, lines=True)
        mapping = {}

        for i in dataset. index:
            http_req = str(dataset["http_req"][i])
            request_id = str(dataset["request_id"][i])

            if http_req not in mapping:
                mapping[http_req] = request_id

        return mapping

    except FileNotFoundError:
        return {}
    except Exception as e:
        logging. error(f"Error loading request mapping:  {e}")
        return {}


def load_tracking_functions(filename: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load tracking functions from features file.

    Args:
        filename:  Path to features.xlsx file

    Returns:
        Dictionary mapping script URL to list of tracking function info
    """
    try:
        dataset = pd.read_excel(filename)
        tracking_functions = {}

        for i in dataset.index:
            # Check if it's a tracking function (label == 1)
            if dataset["label"][i] != 1:
                continue

            script_name = str(dataset["script_name"][i])
            method_name = str(dataset["method_name"][i])

            if script_name not in tracking_functions:
                tracking_functions[script_name] = []

            func_info = {
                "method_name":  method_name,
                "label": int(dataset["label"][i])
            }

            # Add subtype if available
            if "subtype" in dataset. columns:
                func_info["subtype"] = str(dataset["subtype"][i])
            elif "prediction_subtype" in dataset.columns:
                func_info["subtype"] = str(dataset["prediction_subtype"][i])
            else:
                func_info["subtype"] = ""

            tracking_functions[script_name].append(func_info)

        return tracking_functions

    except FileNotFoundError:
        return {}
    except Exception as e:
        logging. error(f"Error loading tracking functions: {e}")
        return {}


def contains_only_numbers(input_string: str) -> bool:
    """
    Check if string contains only numeric characters.

    Used to distinguish between external scripts (numeric request IDs)
    and inline scripts (non-numeric IDs).

    Args:
        input_string: String to check

    Returns:
        True if string is numeric (possibly with decimal point)
    """
    input_string = str(input_string).strip()
    return input_string.replace(".", "", 1).isdigit()


def get_surrogate_function_name(subtype: str) -> str:
    """
    Get the appropriate surrogate function name for a subtype.

    Args:
        subtype: Tracking function subtype

    Returns:
        Name of the surrogate function to use
    """
    surrogate_mapping = {
        "storage_tracking": "surrogateStorage",
        "network_beacon": "surrogateNetworkBeacon",
        "fingerprinting": "surrogateFingerprinting",
        "conversion_analytics": "surrogateConversionAnalytics",
    }
    return surrogate_mapping. get(subtype, "blockme")


# ============================================================================
# Main Processing Functions
# ============================================================================

def process_single_function(
    response_file: str,
    surrogate_file: str,
    method_info: Dict[str, Any],
    script_name: str,
    logger: logging.Logger,
    use_type_aware: bool = True
) -> Tuple[bool, str]:
    """
    Process and replace a single tracking function.

    Args:
        response_file: Path to original script file
        surrogate_file: Path to output surrogate file
        method_info: Method information dictionary
        script_name: Script URL/name
        logger:  Logger instance
        use_type_aware: Whether to use type-aware replacement

    Returns:
        Tuple of (success, error_message)
    """
    method_name = method_info["method_name"]
    subtype = method_info.get("subtype", "")

    # Parse method name for line/column numbers
    # Format: funcName@lineNumber@columnNumber or funcName@lineNumber@columnNumber@stackHash
    parts = method_name. split("@")

    if len(parts) < 3:
        return False, f"Invalid method format: {method_name}"

    try:
        # Line and column are 0-indexed in the data, convert to 1-indexed
        line_num = int(parts[1]) + 1
        column_num = int(parts[2]) + 1
    except ValueError as e:
        return False, f"Could not parse line/column from {method_name}: {e}"

    try:
        logger.debug(
            f"Replacing function at line {line_num}, column {column_num} "
            f"in {script_name} (subtype: {subtype})"
        )

        # Use the function replacer
        status = replace_function_call(
            js_file=response_file,
            modified_js_file=surrogate_file,
            line_number=line_num,
            column_number=column_num
        )

        if status == 0:
            return True, ""
        else:
            return False, "End index not found for function call"

    except Exception as e:
        return False, str(e)


def process_website(
    website_folder:  str,
    logger: logging.Logger,
    use_type_aware: bool = True
) -> Dict[str, Any]:
    """
    Process a single website and generate surrogates for all tracking functions.

    Args:
        website_folder: Path to website output folder
        logger:  Logger instance
        use_type_aware:  Whether to use type-aware surrogates

    Returns:
        Processing results dictionary
    """
    website_name = os.path. basename(website_folder)

    # Initialize results
    results = {
        "website":  website_name,
        "script_not_in_request_file": 0,
        "inline_script": 0,
        "replace_function_call_fail": 0,
        "success": 0,
        "by_subtype":  {},
        "errors":  []
    }

    try:
        # Load request mapping:  {script_url: request_id}
        request_file = os.path.join(website_folder, "request. json")
        request_mapping = load_request_response_mapping(request_file)

        if not request_mapping:
            logger.warning(f"No request mapping found for {website_name}")
            return results

        # Load tracking functions:  {script_url:  [method_info, ...]}
        features_file = os. path.join(website_folder, "features.xlsx")
        tracking_functions = load_tracking_functions(features_file)

        if not tracking_functions:
            logger.info(f"No tracking functions found for {website_name}")
            return results

        # Ensure surrogate directory exists
        surrogate_dir = os.path. join(website_folder, "surrogate")
        os.makedirs(surrogate_dir, exist_ok=True)

        # Process each script with tracking functions
        for script_name, methods in tracking_functions. items():

            # Check if script is in request mapping
            if script_name not in request_mapping:
                for method in methods:
                    results["script_not_in_request_file"] += 1
                    logger.debug(
                        f"Script not in request file: {script_name}, "
                        f"method: {method['method_name']}"
                    )
                continue

            request_id = str(request_mapping[script_name])

            # Check if it's an inline script (non-numeric request ID)
            if not contains_only_numbers(request_id):
                results["inline_script"] += len(methods)
                logger.debug(f"Skipping inline script: {script_name}")
                continue

            # Paths for this script
            response_file = os. path.join(
                website_folder, "response", f"{request_id}.txt"
            )
            surrogate_file = os.path. join(
                surrogate_dir, f"{request_id}_modified.txt"
            )

            # Check if response file exists
            if not os.path. exists(response_file):
                for method in methods:
                    results["replace_function_call_fail"] += 1
                    logger. warning(f"Response file not found: {response_file}")
                continue

            # Process each tracking function in this script
            for method_info in methods:
                success, error = process_single_function(
                    response_file=response_file,
                    surrogate_file=surrogate_file,
                    method_info=method_info,
                    script_name=script_name,
                    logger=logger,
                    use_type_aware=use_type_aware
                )

                if success:
                    results["success"] += 1

                    # Track by subtype
                    subtype = method_info.get("subtype", "unknown")
                    if subtype:
                        results["by_subtype"][subtype] = \
                            results["by_subtype"]. get(subtype, 0) + 1

                    logger. debug(
                        f"Successfully replaced:  {method_info['method_name']} "
                        f"in {script_name}"
                    )
                else:
                    results["replace_function_call_fail"] += 1
                    results["errors"].append(
                        f"{script_name}:{method_info['method_name']}: {error}"
                    )
                    logger.warning(
                        f"Failed to replace {method_info['method_name']} "
                        f"in {script_name}: {error}"
                    )

        # Save request ID mapping for later use
        request_id_file = os. path.join(website_folder, "request_id.json")
        with open(request_id_file, "w") as f:
            json.dump(request_mapping, f, indent=2)

        # Save surrogate generation logs
        logs_file = os.path. join(website_folder, "surrogate_logs.json")
        with open(logs_file, "w") as f:
            # Don't include full error list in per-website logs
            log_results = {k: v for k, v in results.items() if k != "errors"}
            log_results["error_count"] = len(results["errors"])
            json.dump(log_results, f, indent=2)

    except Exception as e:
        logger.error(f"Error processing website {website_name}: {e}")
        results["errors"].append(str(e))

    return results


def process_all_websites(
    config:  SurrogateMainConfig,
    logger: logging.Logger
) -> ProcessingStatistics:
    """
    Process all websites in the input folder.

    Args:
        config: Configuration options
        logger: Logger instance

    Returns:
        Overall processing statistics
    """
    stats = ProcessingStatistics()

    # Get list of websites
    try:
        websites = os.listdir(config.input_folder)
    except FileNotFoundError:
        logger.error(f"Input folder not found:  {config.input_folder}")
        return stats

    stats.total_websites = len(websites)
    logger.info(f"Found {stats.total_websites} websites to process")

    # Process each website
    for website in websites:
        website_path = os.path. join(config.input_folder, website)

        # Skip if not a directory
        if not os. path.isdir(website_path):
            continue

        logger.info(f"Processing website:  {website}")

        try:
            result = process_website(
                website_folder=website_path,
                logger=logger,
                use_type_aware=config.use_type_aware
            )

            # Update statistics
            stats.success += result["success"]
            stats.failed += result["replace_function_call_fail"]
            stats.skipped_inline += result["inline_script"]
            stats.skipped_not_found += result["script_not_in_request_file"]
            stats.total_functions += (
                result["success"] +
                result["replace_function_call_fail"] +
                result["inline_script"] +
                result["script_not_in_request_file"]
            )

            # Merge subtype counts
            for subtype, count in result["by_subtype"]. items():
                stats.by_subtype[subtype] = \
                    stats. by_subtype. get(subtype, 0) + count

            # Track errors
            stats.errors.extend(result. get("errors", []))

            if result["success"] > 0 or result["replace_function_call_fail"] == 0:
                stats.processed_websites += 1
            else:
                stats. failed_websites += 1

            logger.info(
                f"Completed {website}: "
                f"{result['success']} success, "
                f"{result['replace_function_call_fail']} failed, "
                f"{result['inline_script']} inline skipped"
            )

        except Exception as e:
            logger.error(f"Crashed processing website {website}:  {e}")
            stats.failed_websites += 1
            stats.errors.append(f"{website}: {e}")

    return stats


def package_surrogates_for_chrome(
    input_folder: str,
    output_folder: str,
    logger: logging.Logger
) -> Dict[str, int]:
    """
    Package surrogates for Chrome extension deployment.

    Creates a directory structure matching request URLs for
    declarativeNetRequest redirection.

    Args:
        input_folder:  Folder containing processed websites
        output_folder: Output folder for packaged surrogates
        logger: Logger instance

    Returns:
        Packaging statistics
    """
    stats = {"packaged":  0, "failed": 0}

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    websites = os.listdir(input_folder)

    for website in websites:
        website_path = os.path.join(input_folder, website)

        if not os. path.isdir(website_path):
            continue

        try:
            logger.info(f"Packaging surrogates for:  {website}")

            # Load request ID to URL mapping
            request_file = os. path.join(website_path, "request.json")

            try:
                dataset = pd.read_json(request_file, lines=True)
                id_to_url = {}
                for i in dataset.index:
                    req_id = str(dataset["request_id"][i])
                    http_req = str(dataset["http_req"][i])
                    if req_id not in id_to_url:
                        id_to_url[req_id] = http_req
            except Exception as e:
                logger.warning(f"Could not load request mapping for {website}: {e}")
                continue

            # Get surrogate files
            surrogate_dir = os.path.join(website_path, "surrogate")

            if not os.path. exists(surrogate_dir):
                continue

            for surrogate_file in os.listdir(surrogate_dir):
                if not surrogate_file. endswith("_modified.txt"):
                    continue

                # Extract request ID
                request_id = surrogate_file.split("_")[0]

                if request_id not in id_to_url:
                    stats["failed"] += 1
                    continue

                url = id_to_url[request_id]
                source_path = os.path. join(surrogate_dir, surrogate_file)

                try:
                    # Create directory structure from URL
                    # Remove scheme
                    clean_url = url. replace('http://', '').replace('https://', '')

                    # Extract domain and path
                    parts = clean_url. split('/', 1)
                    domain = parts[0]
                    path = parts[1] if len(parts) > 1 else ''

                    # Create full path
                    full_path = os. path.join(output_folder, domain, path)
                    directory, file_name = os. path.split(full_path)

                    if not file_name:
                        file_name = "index.js"

                    # Create directory and copy file
                    os.makedirs(directory, exist_ok=True)
                    shutil.copy(source_path, os.path.join(directory, file_name))

                    stats["packaged"] += 1

                except Exception as e:
                    logger.warning(f"Could not package {surrogate_file}:  {e}")
                    stats["failed"] += 1

        except Exception as e:
            logger. error(f"Error packaging {website}: {e}")

    return stats


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for surrogate generation."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="MTL-TABlock Surrogate Script Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples: 
    # Process all websites in default folder
    python surrogate_main.py
    
    # Process specific input/output folders
    python surrogate_main.py --input server/output --output server/surrogates
    
    # Process single website
    python surrogate_main.py --website example.com
    
    # Package for Chrome extension
    python surrogate_main. py --package-chrome
        """
    )

    parser. add_argument(
        "--input", "-i",
        type=str,
        default="server/output",
        help="Input folder containing website data (default: server/output)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="server/surrogates",
        help="Output folder for packaged surrogates (default: server/surrogates)"
    )

    parser.add_argument(
        "--website", "-w",
        type=str,
        default=None,
        help="Process only this specific website"
    )

    parser.add_argument(
        "--type-aware",
        action="store_true",
        default=True,
        help="Use type-aware surrogate replacement (default: True)"
    )

    parser.add_argument(
        "--package-chrome",
        action="store_true",
        help="Package surrogates for Chrome extension deployment"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="function_logs.json",
        help="Log file name (default: function_logs.json)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser. parse_args()

    # Setup configuration
    config = SurrogateMainConfig(
        input_folder=args.input,
        output_folder=args.output,
        use_type_aware=args.type_aware,
        log_level=logging.DEBUG if args.verbose else logging.INFO
    )

    # Setup logging
    logger = setup_logging(
        log_folder=config.logs_folder,
        log_file=args. log_file,
        level=config. log_level
    )

    logger.info("=" * 60)
    logger.info("MTL-TABlock Surrogate Script Generator")
    logger.info("=" * 60)
    logger.info(f"Input folder: {config. input_folder}")
    logger.info(f"Output folder:  {config.output_folder}")
    logger.info(f"Type-aware replacement: {config.use_type_aware}")

    # Process websites
    if args.website:
        # Process single website
        website_path = os. path.join(config.input_folder, args.website)

        if not os.path.exists(website_path):
            logger.error(f"Website folder not found: {website_path}")
            sys.exit(1)

        logger.info(f"Processing single website: {args. website}")

        result = process_website(
            website_folder=website_path,
            logger=logger,
            use_type_aware=config.use_type_aware
        )

        logger.info(f"Results: {json.dumps(result, indent=2)}")

    else:
        # Process all websites
        logger.info("Processing all websites...")

        stats = process_all_websites(config, logger)

        # Log summary
        logger. info("=" * 60)
        logger.info("SURROGATE GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total websites:  {stats.total_websites}")
        logger.info(f"Processed websites: {stats. processed_websites}")
        logger.info(f"Failed websites:  {stats.failed_websites}")
        logger.info(f"Total functions: {stats. total_functions}")
        logger.info(f"Successfully replaced: {stats.success}")
        logger.info(f"Failed to replace: {stats. failed}")
        logger.info(f"Skipped (inline): {stats.skipped_inline}")
        logger.info(f"Skipped (not found): {stats.skipped_not_found}")

        if stats.by_subtype:
            logger.info("By subtype:")
            for subtype, count in stats. by_subtype. items():
                logger.info(f"  {subtype}:  {count}")

        # Calculate success rate
        replaceable = stats.total_functions - stats.skipped_inline - stats.skipped_not_found
        if replaceable > 0:
            success_rate = stats.success / replaceable * 100
            logger.info(f"Success rate: {success_rate:.1f}%")

        # Save overall statistics
        if config.save_statistics:
            stats_file = os. path.join(config.logs_folder, "generation_stats.json")
            os.makedirs(config.logs_folder, exist_ok=True)
            with open(stats_file, "w") as f:
                json. dump(stats. to_dict(), f, indent=2)
            logger.info(f"Statistics saved to:  {stats_file}")

    # Package for Chrome if requested
    if args.package_chrome:
        logger.info("=" * 60)
        logger.info("Packaging surrogates for Chrome extension...")

        package_stats = package_surrogates_for_chrome(
            input_folder=config.input_folder,
            output_folder=config.output_folder,
            logger=logger
        )

        logger.info(f"Packaged:  {package_stats['packaged']}")
        logger.info(f"Failed: {package_stats['failed']}")

    logger.info("=" * 60)
    logger.info("Surrogate generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()