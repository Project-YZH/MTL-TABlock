"""
MTL-TABlock:  Surrogate Script Generator

"""

import os
import json
import logging
import shutil
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import hashlib

from .utils.function_replacer import (
    FunctionReplacer,
    FunctionLocation,
    ReplacementStrategy,
    ReplacementResult
)
from .surrogate_templates. storage_tracking_surrogate import (
    StorageTrackingSurrogateGenerator,
    get_storage_surrogate_template
)
from .surrogate_templates. network_beacon_surrogate import (
    NetworkBeaconSurrogateGenerator,
    get_network_beacon_surrogate_template
)
from .surrogate_templates.fingerprinting_surrogate import (
    FingerprintingSurrogateGenerator,
    get_fingerprinting_surrogate_template
)
from .surrogate_templates.conversion_analytics_surrogate import (
    ConversionAnalyticsSurrogateGenerator,
    get_conversion_analytics_surrogate_template
)


@dataclass
class SurrogateGenerationConfig:
    """Configuration for surrogate generation."""
    # Input/Output paths
    output_folder: str = "server/output"
    surrogates_folder:  str = "server/surrogates"
    logs_folder: str = "logs"
    
    # Generation options
    replacement_strategy: ReplacementStrategy = ReplacementStrategy.TYPE_AWARE
    preserve_length: bool = True
    include_infrastructure: bool = True
    minify_output: bool = False
    
    # Subtype handling
    use_type_aware_surrogates: bool = True
    fallback_to_blockme: bool = True
    
    # Logging
    log_level: int = logging.INFO
    log_to_file: bool = True


@dataclass
class GenerationStatistics:
    """Statistics for surrogate generation."""
    total_scripts: int = 0
    processed_scripts: int = 0
    failed_scripts: int = 0
    
    total_functions: int = 0
    replaced_functions: int = 0
    failed_functions: int = 0
    skipped_functions: int = 0
    
    by_subtype: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_scripts": self. total_scripts,
            "processed_scripts": self.processed_scripts,
            "failed_scripts": self.failed_scripts,
            "total_functions": self.total_functions,
            "replaced_functions":  self.replaced_functions,
            "failed_functions": self. failed_functions,
            "skipped_functions": self.skipped_functions,
            "by_subtype": self.by_subtype,
            "success_rate": self. replaced_functions / max(1, self.total_functions),
            "error_count": len(self.errors)
        }


class SurrogateGenerator:
    """
    Main class for generating surrogate scripts.
    
    This class orchestrates the entire surrogate generation pipeline: 
    1. Load predictions from the MTL model
    2. Match predictions to script files
    3. Generate type-aware surrogates for each tracking function
    4. Package surrogates for deployment
    """
    
    def __init__(self, config: SurrogateGenerationConfig = None):
        """
        Initialize the surrogate generator. 
        
        Args: 
            config: Configuration options
        """
        self.config = config or SurrogateGenerationConfig()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize subtype generators
        self. storage_generator = StorageTrackingSurrogateGenerator()
        self.beacon_generator = NetworkBeaconSurrogateGenerator()
        self.fingerprint_generator = FingerprintingSurrogateGenerator()
        self.conversion_generator = ConversionAnalyticsSurrogateGenerator()
        
        # Initialize function replacer
        self.replacer = FunctionReplacer(
            strategy=self.config.replacement_strategy,
            preserve_length=self.config. preserve_length,
            logger=self. logger
        )
        
        # Statistics
        self.stats = GenerationStatistics()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger("SurrogateGenerator")
        self.logger.setLevel(self.config. log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config. log_level)
        console_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file:
            os.makedirs(self.config.logs_folder, exist_ok=True)
            log_file = os. path.join(
                self.config. logs_folder,
                f"surrogate_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler. setLevel(logging.DEBUG)
            file_handler.setFormatter(console_format)
            self.logger.addHandler(file_handler)
    
    def generate_for_website(self, website_folder: str) -> Dict[str, Any]: 
        """
        Generate surrogates for a single website.
        
        Args:
            website_folder: Path to the website's output folder
            
        Returns:
            Dictionary with generation results
        """
        self.logger.info(f"Starting surrogate generation for:  {website_folder}")
        
        results = {
            "website":  os.path.basename(website_folder),
            "success": False,
            "functions_replaced": 0,
            "functions_failed": 0,
            "scripts_modified": 0,
            "errors": []
        }
        
        try:
            # Load request ID mapping
            request_mapping = self._load_request_mapping(website_folder)
            if not request_mapping: 
                results["errors"].append("Could not load request mapping")
                return results
            
            # Load tracking functions from features
            tracking_functions = self._load_tracking_functions(website_folder)
            if not tracking_functions:
                results["errors"].append("No tracking functions found")
                return results
            
            # Generate infrastructure templates
            if self.config.include_infrastructure:
                self._generate_infrastructure(website_folder)
            
            # Process each script
            for script_url, functions in tracking_functions.items():
                script_result = self._process_script(
                    website_folder,
                    script_url,
                    functions,
                    request_mapping
                )
                
                results["functions_replaced"] += script_result["replaced"]
                results["functions_failed"] += script_result["failed"]
                
                if script_result["replaced"] > 0:
                    results["scripts_modified"] += 1
            
            results["success"] = results["functions_failed"] == 0 or results["functions_replaced"] > 0
            
            # Save logs
            self._save_website_logs(website_folder, results)
            
            self. logger.info(
                f"Completed {website_folder}:  "
                f"{results['functions_replaced']} replaced, "
                f"{results['functions_failed']} failed"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {website_folder}: {e}")
            results["errors"].append(str(e))
        
        return results
    
    def generate_for_all_websites(self) -> GenerationStatistics: 
        """
        Generate surrogates for all websites in the output folder.
        
        Returns:
            Overall generation statistics
        """
        self.logger. info("Starting batch surrogate generation")
        
        websites = os.listdir(self.config.output_folder)
        self.stats.total_scripts = len(websites)
        
        for website in websites: 
            website_path = os.path.join(self.config. output_folder, website)
            
            if not os. path.isdir(website_path):
                continue
            
            try:
                result = self.generate_for_website(website_path)
                
                if result["success"]: 
                    self. stats.processed_scripts += 1
                else:
                    self.stats.failed_scripts += 1
                
                self. stats.replaced_functions += result["functions_replaced"]
                self.stats. failed_functions += result["functions_failed"]
                
            except Exception as e: 
                self.logger.error(f"Failed to process website {website}: {e}")
                self. stats.failed_scripts += 1
                self.stats.errors. append(f"{website}: {e}")
        
        # Save overall statistics
        self._save_overall_statistics()
        
        self.logger.info(
            f"Batch generation complete: "
            f"{self. stats.processed_scripts}/{self.stats.total_scripts} websites, "
            f"{self.stats.replaced_functions} functions replaced"
        )
        
        return self.stats
    
    def _load_request_mapping(self, website_folder: str) -> Dict[str, str]:
        """Load mapping from script URL to request ID."""
        import pandas as pd
        
        request_file = os.path. join(website_folder, "request. json")
        
        if not os. path.exists(request_file):
            return {}
        
        try:
            dataset = pd.read_json(request_file, lines=True)
            mapping = {}
            
            for i in dataset.index:
                http_req = dataset["http_req"][i]
                request_id = dataset["request_id"][i]
                
                if http_req not in mapping:
                    mapping[http_req] = request_id
            
            return mapping
            
        except Exception as e:
            self.logger.error(f"Error loading request mapping: {e}")
            return {}
    
    def _load_tracking_functions(
        self,
        website_folder: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load tracking functions from features file.
        
        Returns:
            Dictionary mapping script URLs to list of tracking function info
        """
        import pandas as pd
        
        features_file = os.path.join(website_folder, "features. xlsx")
        
        if not os.path.exists(features_file):
            return {}
        
        try:
            dataset = pd.read_excel(features_file)
            tracking_functions = {}
            
            for i in dataset.index:
                # Check if it's a tracking function (label == 1)
                if dataset["label"][i] != 1:
                    continue
                
                script_name = dataset["script_name"][i]
                method_name = dataset["method_name"][i]
                
                # Get subtype if available
                subtype = ""
                if "subtype" in dataset. columns:
                    subtype = dataset["subtype"][i]
                elif "prediction_subtype" in dataset.columns:
                    subtype = dataset["prediction_subtype"][i]
                
                if script_name not in tracking_functions: 
                    tracking_functions[script_name] = []
                
                tracking_functions[script_name].append({
                    "method_name": method_name,
                    "subtype": subtype,
                    "label": dataset["label"][i]
                })
            
            return tracking_functions
            
        except Exception as e:
            self.logger.error(f"Error loading tracking functions: {e}")
            return {}
    
    def _process_script(
        self,
        website_folder: str,
        script_url: str,
        functions: List[Dict[str, Any]],
        request_mapping: Dict[str, str]
    ) -> Dict[str, int]:
        """
        Process a single script and replace its tracking functions.
        
        Args:
            website_folder: Website folder path
            script_url: URL of the script
            functions: List of tracking functions in the script
            request_mapping: URL to request ID mapping
            
        Returns: 
            Dictionary with replaced/failed counts
        """
        result = {"replaced": 0, "failed": 0, "skipped": 0}
        
        # Get request ID for this script
        if script_url not in request_mapping: 
            self.logger.warning(f"Script not in request mapping: {script_url}")
            result["skipped"] = len(functions)
            return result
        
        request_id = str(request_mapping[script_url])
        
        # Check if it's an inline script
        if not request_id.replace(".", "").isdigit():
            self.logger.info(f"Skipping inline script: {script_url}")
            result["skipped"] = len(functions)
            return result
        
        # Paths
        response_file = os. path.join(website_folder, "response", f"{request_id}.txt")
        surrogate_dir = os.path. join(website_folder, "surrogate")
        surrogate_file = os.path.join(surrogate_dir, f"{request_id}_modified.txt")
        
        # Ensure surrogate directory exists
        os.makedirs(surrogate_dir, exist_ok=True)
        
        # Check if response file exists
        if not os.path. exists(response_file):
            self.logger.warning(f"Response file not found: {response_file}")
            result["failed"] = len(functions)
            return result
        
        # Process each function
        for func_info in functions: 
            method_name = func_info["method_name"]
            subtype = func_info. get("subtype", "")
            
            # Parse method name for line/column numbers
            parts = method_name. split("@")
            if len(parts) < 3:
                self.logger.warning(f"Invalid method format: {method_name}")
                result["failed"] += 1
                continue
            
            try:
                line_num = int(parts[1]) + 1
                column_num = int(parts[2]) + 1
                
                # Replace the function
                status = self.replacer.replace_function_call(
                    js_file=response_file,
                    modified_js_file=surrogate_file,
                    line_number=line_num,
                    column_number=column_num,
                    subtype=subtype,
                    function_name=parts[0]
                )
                
                if status == 0:
                    result["replaced"] += 1
                    
                    # Update subtype statistics
                    if subtype: 
                        self. stats.by_subtype[subtype] = \
                            self. stats.by_subtype.get(subtype, 0) + 1
                else:
                    result["failed"] += 1
                    
            except Exception as e:
                self. logger.error(f"Error replacing {method_name}: {e}")
                result["failed"] += 1
        
        return result
    
    def _generate_infrastructure(self, website_folder: str) -> None:
        """Generate surrogate infrastructure templates."""
        surrogate_dir = os.path.join(website_folder, "surrogate")
        os.makedirs(surrogate_dir, exist_ok=True)
        
        # Combine all infrastructure templates
        infrastructure = "// MTL-TABlock Surrogate Infrastructure\n\n"
        infrastructure += get_storage_surrogate_template() + "\n\n"
        infrastructure += get_network_beacon_surrogate_template() + "\n\n"
        infrastructure += get_fingerprinting_surrogate_template() + "\n\n"
        infrastructure += get_conversion_analytics_surrogate_template() + "\n"
        
        # Save infrastructure file
        infra_file = os.path.join(surrogate_dir, "_mtl_infrastructure.js")
        with open(infra_file, "w", encoding="utf-8") as f:
            f. write(infrastructure)
    
    def _save_website_logs(self, website_folder: str, results:  Dict[str, Any]) -> None:
        """Save generation logs for a website."""
        logs_file = os.path. join(website_folder, "surrogate_logs.json")
        
        with open(logs_file, "w") as f:
            json.dump(results, f, indent=2)
    
    def _save_overall_statistics(self) -> None:
        """Save overall generation statistics."""
        stats_file = os. path.join(
            self.config. logs_folder,
            "generation_statistics. json"
        )
        
        os.makedirs(self.config.logs_folder, exist_ok=True)
        
        with open(stats_file, "w") as f:
            json.dump(self.stats. to_dict(), f, indent=2)
    
    def get_statistics(self) -> GenerationStatistics: 
        """Get current generation statistics."""
        return self.stats


class ChromeSurrogatePackager:
    """
    Packages surrogates for Chrome extension deployment.
    
    Creates a directory structure matching request URLs for
    declarativeNetRequest redirection.
    """
    
    def __init__(
        self,
        output_folder: str = "server/output",
        surrogates_folder: str = "server/surrogates"
    ):
        """
        Initialize the packager.
        
        Args:
            output_folder:  Folder containing processed websites
            surrogates_folder: Output folder for packaged surrogates
        """
        self.output_folder = output_folder
        self.surrogates_folder = surrogates_folder
        self.logger = logging.getLogger("ChromeSurrogatePackager")
    
    def _load_request_mapping(self, request_file: str) -> Dict[str, str]: 
        """Load request ID to URL mapping."""
        import pandas as pd
        
        try:
            dataset = pd.read_json(request_file, lines=True)
            mapping = {}
            
            for i in dataset.index:
                request_id = dataset["request_id"][i]
                http_req = dataset["http_req"][i]
                
                if request_id not in mapping:
                    mapping[request_id] = http_req
            
            return mapping
            
        except Exception as e: 
            self.logger. error(f"Error loading request mapping: {e}")
            return {}
    
    def _create_directory_from_url(
        self,
        url: str,
        folder_path: str,
        source_file: str
    ) -> bool:
        """
        Create directory structure from URL and copy surrogate file.
        
        Args:
            url: Request URL
            folder_path: Base folder for surrogates
            source_file: Path to the surrogate file
            
        Returns: 
            True on success, False on failure
        """
        try: 
            # Remove scheme
            url = url. replace('http://', '').replace('https://', '')
            
            # Extract domain and path
            parts = url.split('/', 1)
            domain = parts[0]
            path = parts[1] if len(parts) > 1 else ''
            
            # Create full path
            full_path = os.path.join(folder_path, domain, path)
            directory, file_name = os.path.split(full_path)
            
            # Ensure file name exists
            if not file_name: 
                file_name = "index.js"
            
            # Create directory
            os.makedirs(directory, exist_ok=True)
            
            # Copy file
            shutil. copy(source_file, os.path.join(directory, file_name))
            
            return True
            
        except Exception as e: 
            self.logger.error(f"Error creating directory structure: {e}")
            return False
    
    def package_website(self, website_name: str) -> Dict[str, Any]: 
        """
        Package surrogates for a single website.
        
        Args:
            website_name: Name of the website folder
            
        Returns:
            Packaging results
        """
        results = {
            "website": website_name,
            "packaged": 0,
            "failed": 0,
            "errors": []
        }
        
        website_folder = os. path.join(self.output_folder, website_name)
        
        try:
            # Load request mapping
            request_file = os.path. join(website_folder, "request.json")
            request_mapping = self._load_request_mapping(request_file)
            
            if not request_mapping: 
                results["errors"]. append("No request mapping found")
                return results
            
            # Get surrogate files
            surrogate_folder = os.path. join(website_folder, "surrogate")
            
            if not os. path.exists(surrogate_folder):
                results["errors"]. append("No surrogate folder found")
                return results
            
            for surrogate_file in os.listdir(surrogate_folder):
                if not surrogate_file. endswith("_modified.txt"):
                    continue
                
                # Extract request ID
                request_id = surrogate_file.split("_")[0]
                
                # Get URL for this request
                if request_id not in request_mapping: 
                    results["failed"] += 1
                    continue
                
                url = request_mapping[request_id]
                source_path = os.path. join(surrogate_folder, surrogate_file)
                
                # Create directory structure and copy
                success = self._create_directory_from_url(
                    url,
                    self.surrogates_folder,
                    source_path
                )
                
                if success:
                    results["packaged"] += 1
                else:
                    results["failed"] += 1
                    
        except Exception as e: 
            self.logger.error(f"Error packaging {website_name}: {e}")
            results["errors"].append(str(e))
        
        return results
    
    def package_all(self) -> Dict[str, Any]: 
        """
        Package surrogates for all websites.
        
        Returns:
            Overall packaging results
        """
        self.logger.info("Starting surrogate packaging for Chrome")
        
        overall_results = {
            "total_websites": 0,
            "successful_websites": 0,
            "total_packaged": 0,
            "total_failed": 0
        }
        
        # Ensure output folder exists
        os.makedirs(self.surrogates_folder, exist_ok=True)
        
        websites = os.listdir(self.output_folder)
        overall_results["total_websites"] = len(websites)
        
        for website in websites:
            website_path = os.path.join(self. output_folder, website)
            
            if not os.path.isdir(website_path):
                continue
            
            self.logger.info(f"Packaging surrogates for:  {website}")
            
            result = self.package_website(website)
            
            if result["packaged"] > 0:
                overall_results["successful_websites"] += 1
            
            overall_results["total_packaged"] += result["packaged"]
            overall_results["total_failed"] += result["failed"]
        
        self.logger.info(
            f"Packaging complete:  {overall_results['total_packaged']} files packaged"
        )
        
        return overall_results