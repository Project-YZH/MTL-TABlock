# MTL-TABlock

**MTL-TABlock: Function-level Type-aware Tracking Script Blocking via Multi-task Learning**

MTL-TABlock addresses the privacy–usability tension caused by *mixed scripts*: a single script often contains both tracking logic and essential site functionality, so script-level or domain-level blocking can easily break pages.  
This project operates at **function granularity**. It builds function-level behavioral graphs from browser runtime signals and trains a **multi-task learning (MTL)** model to jointly perform:

1) **Tracking function detection** (tracking vs. benign)  
2) **Tracking function subtype identification** (type-aware), covering: **Storage Tracking / Network Beacon / Fingerprinting / Conversion Analytics**

Based on the predicted subtype, MTL-TABlock generates and injects **type-aware surrogates** (compatible replacement functions) to block tracking behavior while preserving page functionality as much as possible.

> Note: This repository is a research / reproduction-oriented prototype. 

---

## Table of Contents

- [Key Features](#key-features)
- [Method Overview](#method-overview)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Surrogate Strategies](#surrogate-strategies)
- [Extension Injection Approaches](#extension-injection-approaches)
- [Data & Compliance](#data--compliance)
- [Citation](#citation)
- [License](#license)

---

## Key Features

- **Function-level Behavior Graphs**: Captures network requests, DOM mutations, storage accesses, key Web API calls, and call stacks to build fine-grained graph representations.
- **Structural + Contextual Feature Fusion**: Encodes both graph structure (call/interaction relations) and runtime semantics.
- **Multi-Task Learning (MTL)**: A primary task for tracking detection plus an auxiliary task for subtype classification to drive downstream, subtype-specific blocking.
- **Type-aware Surrogate Generation**: Produces interface-compatible replacements (preserving return shapes and sync/async semantics) to reduce breakage versus naïve “no-op” stubs.

---

## Method Overview

The overall pipeline consists of six stages:

1. **Data Collection**: Chrome extension + automated crawling to collect runtime signals (network/DOM/storage/WebAPI/call stack).
2. **Graph Construction**: Build site/script-level behavior graphs (function nodes, network nodes, storage nodes, WebAPI nodes, and interaction edges).
3. **Feature Extraction**: Compute structural and contextual features to form function-level samples.
4. **Function Annotation**: Derive training labels (tracking/benign + subtype) using filter lists (EasyList/EasyPrivacy) and high-confidence rules.
5. **Model Training**: Train the MTL model (tracking detection + subtype classification).
6. **Surrogate Generation & Deployment**: Identify target functions and inject type-aware surrogates at runtime.

---

## Repository Layout

> The layout below follows the project’s stage-based organization.

```
.
├── 1_data_collection
│   ├── browser_extension
│   │   ├── manifest.json
│   │   ├── background.js
│   │   ├── content.js
│   │   ├── inject.js
│   │   ├── basic.html
│   │   └── breakpoint.json
│   ├── data_collection_server
│   │   ├── package.json
│   │   ├── package-lock.json
│   │   └── server.js
│   └── selenium_crawler
│       ├── crawler_main.py
│       └── crawler_with_hook.py
├── 2_graph_construction
│   ├── graph_builder_main.py
│   ├── graph_population.py
│   ├── graph_population_with_callstack.py
│   └── node_handlers
│       ├── event_handler.py
│       ├── info_share_handler.py
│       ├── network_node_handler.py
│       ├── redirection_edge_handler.py
│       └── storage_node_handler.py
├── 3_feature_extraction
│   ├── feature_extractor_main.py
│   ├── contextual_features.py
│   ├── structural_features.py
│   ├── network_features.py
│   └── network_features_methods.py
├── 4_function_annotation
│   ├── tracking_annotation.py
│   ├── subtype_annotation.py
│   └── filter_lists
│       ├── easylist_parser.py
│       └── high_confidence_rules.py
├── 5_model_training
│   ├── model_main.py
│   └── mtl_model.py
├── 6_surrogate_generation
│   ├── surrogate_main.py
│   ├── surrogate_generator.py
│   ├── function_replacer.py
│   ├── parentheses_balance.py
│   └── surrogate_templates
│       ├── storage_tracking_surrogate.py
│       ├── network_beacon_surrogate.py
│       ├── fingerprinting_surrogate.py
│       └── conversion_analytics_surrogate.py
├── utils
│   └── common_utils.py
└── requirements.txt
```

---

## Requirements

### Recommended Environment

- **Python**: 3.8+ (recommended 3.10+)
- **Node.js**: 16+ (for `data_collection_server`)
- **Selenium + matching ChromeDriver**

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Node Dependencies (Collection Server)

```bash
cd 1_data_collection/data_collection_server
npm install
```

---

## Quick Start

### 1) Start the Data Collection Server (Node)

```bash
cd 1_data_collection/data_collection_server
node server.js
```

### 2) Load the Chrome Extension (Developer Mode)

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select: `1_data_collection/browser_extension/`

### 3) Run the Selenium Crawler

```bash
cd 1_data_collection/selenium_crawler
python crawler_main.py 
```

For stronger instrumentation/hooking (if available in your environment):

```bash
python crawler_with_hook.py 
```

### 4) Graph Construction

```bash
cd 2_graph_construction
python graph_builder_main.py 
```

### 5) Feature Extraction

```bash
cd 3_feature_extraction
python feature_extractor_main.py 
```

### 6) Function Annotation (Tracking / Subtype)

```bash
cd 4_function_annotation
python tracking_annotation.py 
python subtype_annotation.py 
```

### 7) MTL Model Training

```bash
cd 5_model_training
python model_main.py 
```

### 8) Surrogate Generation

```bash
cd 6_surrogate_generation
python surrogate_main.py 
```

---

## Surrogate Strategies

Surrogate templates are under `6_surrogate_generation/surrogate_templates/`:

- **Storage Tracking**: Replace cross-site linkable identifiers with site-scoped pseudo-identifiers to reduce cross-site correlation while keeping site behavior stable.
- **Network Beacon**: Suppress actual exfiltration while returning “success” semantics (e.g., resolved Promises) to preserve control flow.
- **Fingerprinting**: Return low-entropy, origin-stable, cross-origin-unlinkable pseudo-fingerprints.
- **Conversion Analytics**: Locally emulate assignment/variant logic and return stable, business-compatible results without contacting remote endpoints.

---

## Extension Injection Approaches

Two injection approaches are described:

- **Manifest V2 (MV2)**: Use Chrome DevTools Protocol (CDP) to intercept scripts before execution and replace them with surrogates.
- **Manifest V3 (MV3)**: Use `declarativeNetRequest` to redirect target script requests to locally packaged surrogate files (MV3 cannot directly rewrite response bodies in the same way as MV2).

The directory `1_data_collection/browser_extension/` can serve as the starting point for extension-based instrumentation and injection logic.

---

## Data & Compliance

Dynamic collection involves scripts, network requests, and runtime context. Please ensure that you:

- Collect data only under lawful conditions (authorization / academic research / compliant testing);
- Avoid collecting or storing sensitive personal data;
- Apply anonymization and access control before releasing datasets/models.

---

## Citation

If this project is helpful for your research, please cite the paper:

```bibtex
@article{mtl_tablock,
  title   = {MTL-TABlock: Function-level Type-aware Tracking Script Blocking via Multi-task Learning},
  author  = {Zhanhui Yuan and Zhi Yang and Jinglei Tan and Hao Hu and Hongqi Zhang},
  note    = {See MTL-TABlock.md in this repository for the full text},
}
```

---

## License

Add your preferred open-source license (e.g., MIT / Apache-2.0 / GPL-3.0) and any third-party notices (EasyList/EasyPrivacy, etc.) before publishing.
