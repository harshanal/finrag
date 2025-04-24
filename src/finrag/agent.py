"""
########################################################################
# FinRAG Agent Module (agent.py)
#
# This module implements the "planning and execution" stage of the RAG pipeline
# using a two-step LLM approach:
# 1. Specify Expression (Step 1): Takes the question and retrieved evidence chunks.
#    - Calls an LLM (specify_and_generate_expression) to determine the required
#      financial items, map them to variables (e.g., VAL_1), and generate a 
#      Python expression template (e.g., "(VAL_1 - VAL_2) / VAL_2") to answer the question.
#      It also determines the required output format/units.
# 2. Extract Values (Step 2): Takes the required items map from Step 1 and evidence.
#    - Calls an LLM (extract_all_values_with_llm) to find the numerical values for
#      each required item within the provided evidence chunks.
# 3. Execute: Substitutes the extracted values into the Python expression template
#    and evaluates it to compute the final answer. Handles scaling and formatting.
#
# Main functions:
# - specify_and_generate_expression(question, evidence_chunks): 
#     Performs Step 1: generates the calculation plan (required items, expression template).
# - extract_all_values_with_llm(required_items_map, evidence_chunks):
#     Performs Step 2: extracts numerical values from evidence based on the plan.
# - plan_and_execute(question, evidence_chunks):
#     Orchestrates the full process: calls Step 1, then Step 2, executes the 
#     expression, handles errors (e.g., extraction failures), and returns a 
#     structured dict containing program, intermediates, answer, tool used, and evidence IDs.
########################################################################
"""

"""Agent module for FinRAG: specify_and_express -> extract_all -> substitute_and_eval -> format."""

import json
import openai
import warnings
import re
import os
from typing import Any, Dict, List, Tuple, Optional
import logging
import math # Needed for safe eval context

# Configure logger
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def clean_numeric_string(s: str) -> str:
    """Clean a string potentially representing a number to a pure numeric format."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip() # Ensure input is string and stripped

    # --- START: Pre-cleaning: Remove common units/qualifiers (case-insensitive) ---
    s_lower = s.lower()
    # Remove currency symbols first (before checking for 'usd', etc.)
    s = s.replace('$', '').replace('£', '').replace('€', '') # Added common currency symbols

    # Words/Units to remove 
    # Keep this list concise - extraction prompt should handle most textual units
    words_to_remove = [
        "approx", "approximately", "around", "about",
        "usd", "eur", "gbp",
        "years", "year",
        "million", "billion", "thousand", "trillion" # Keep these for safety
    ]
    # Remove whole words using regex word boundaries to avoid partial matches
    for word in words_to_remove:
        s = re.sub(r'\b' + re.escape(word) + r'\b', '', s, flags=re.IGNORECASE)

    # After removing words/symbols, strip whitespace again
    s = s.strip()
    # --- END: Pre-cleaning ---

    # Handle parentheses for negative numbers: e.g., (500) -> -500
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    
    # Remove remaining commas 
    s = s.replace(',', '')
    
    # Handle percentage sign: e.g., 15% -> 0.15
    if s.endswith('%'):
        # Ensure the part before % is potentially numeric before converting
        num_part = s[:-1].strip()
        try:
            # Attempt to convert the part before '%' to float and divide by 100
            return str(float(num_part) / 100.0)
        except ValueError:
             # If conversion fails, just return the cleaned numeric part without %
             return num_part
             
    return s

def parse_markdown_table(md_table: str) -> List[Dict[str, str]] | None:
    """Parses a simple Markdown table string into a list of dictionaries."""
    lines = md_table.strip().split('\n')
    if len(lines) < 3: # Need header, separator, and at least one body row
        return None

    # Helper to clean cell content
    def clean_cell(cell: str) -> str:
        return cell.strip().replace('\\\\|', '') # Remove escaped pipes and strip whitespace

    # Process header
    header_line = lines[0].strip()
    if not header_line.startswith('|') or not header_line.endswith('|'):
        return None
    headers = [clean_cell(h) for h in header_line[1:-1].split('|')]

    # Skip separator line (lines[1])

    # Process body rows
    parsed_rows = []
    for line in lines[2:]:
        line = line.strip()
        if not line.startswith('|') or not line.endswith('|'):
            continue # Skip invalid lines
        
        cells = [clean_cell(c) for c in line[1:-1].split('|')]
        if len(cells) == len(headers):
            row_dict = {headers[i]: cells[i] for i in range(len(headers))}
            parsed_rows.append(row_dict)
        else:
             logger.warning(f"Skipping table row due to column mismatch: {line}")

    return parsed_rows if parsed_rows else None

# --- Combined Step 1: Specify Requirements & Generate Expression Template ---

COMBINED_STEP_FUNCTION_NAME = "specify_extract_and_express"
COMBINED_STEP_FUNCTION_SCHEMA = {
    "name": COMBINED_STEP_FUNCTION_NAME,
    "description": "Specifies calculation type, maps required items to placeholders, defines the python expression template, and sets the desired output format.",
        "parameters": {
            "type": "object",
            "properties": {
            "calculation_type": {
                "type": "string",
                "description": "Type of calculation (e.g., 'percentage_change', 'sum', 'average', 'ratio', 'difference', 'value_lookup').",
                "enum": ["percentage_change", "sum", "average", "ratio", "difference", "value_lookup", "other"]
            },
            "required_items_map": {
                "type": "object",
                "description": "Maps generic placeholders (e.g., VAL_1) to specific data item descriptions (e.g., 'Net Sales 2021'). Keys must be VAL_1, VAL_2, etc.",
                "additionalProperties": {
                    "type": "string",
                    "description": "Specific data item description (e.g., 'Metric Name YYYY')"
                }
            },
            "python_expression_template": {
                "type": "string",
                "description": "Single-line Python math expression using placeholder keys (VAL_1, VAL_2...) from required_items_map. Example: '(VAL_1 - VAL_2) / VAL_2'"
            },
            "output_format": {
                 "type": "string",
                 "description": "Desired format for the final answer ('number' or 'percent').",
                 "enum": ["number", "percent"],
                 "default": "number"
             }
        },
        "required": ["calculation_type", "required_items_map", "python_expression_template"]
        # output_format is optional, defaults to number
    }
}

def specify_and_generate_expression(
    question: str,
    evidence_chunks: List[Dict[str, str]]
) -> Optional[Dict[str, Any]]:
    """
    Uses a single LLM call to:
    1. Specify calculation type.
    2. Define required data items and map them to placeholders (VAL_1, VAL_2...). 
    3. Generate a Python expression template using these placeholders.
    4. Specify the final output format.
    Returns a dictionary containing the extracted arguments or None on failure.
    """
    logger.info("Running combined Step 1: Specify requirements and generate expression template...")
    
    # --- Evidence pre-processing ---
    processed_chunks_for_prompt = []
    MAX_CHUNKS_FOR_COMBINED_STEP = 10 
    for i, chunk in enumerate(evidence_chunks[:MAX_CHUNKS_FOR_COMBINED_STEP]):
        chunk_id = chunk.get("chunk_id", f"unknown_{i}")
        text = chunk.get("text", "")
        prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\\n{text}\\n---" # Default format
        is_table_chunk = chunk_id.endswith("::table:full") and text.strip().startswith("Financial Table Data:")
        if is_table_chunk:
            md_content = text.split("\\n\\n", 1)[-1]
            parsed_table = parse_markdown_table(md_content)
            if parsed_table:
                try:
                    json_table_string = json.dumps(parsed_table, separators=(',', ':')) 
                    prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\\n```json\\n{json_table_string}\\n```\\n---"
                    logger.debug(f"Parsed table chunk {chunk_id} into compact JSON.")
                except Exception as json_err:
                    logger.error(f"Failed JSON dump for {chunk_id}: {json_err}")
                    prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\\n{text}\\n---" # Fallback
            else:
                logger.warning(f"Failed table parse for {chunk_id}.")
                prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\\n{text}\\n---" # Fallback
        processed_chunks_for_prompt.append(prompt_text)
    evidence_text = "\\n".join(processed_chunks_for_prompt)
    # --- End Evidence pre-processing ---

    # --- System Prompt for Combined Task ---
    system_prompt = f"""\
You are a meticulous financial analyst assistant. Your task is to analyze a question and evidence, then define the calculation needed by specifying the type, required items mapped to placeholders, a Python expression template, and the desired output format.

**Your Task:**

1.  **Identify Calculation Type:** Determine the core operation (percentage_change, sum, average, ratio, difference, value_lookup).
2.  **Identify Required Data Items & Map to Placeholders:** List *all* specific numerical data points needed. Format each as "Metric Name [Optional Context] YYYY/Period". Create a mapping where keys are simple placeholders (`VAL_1`, `VAL_2`, etc., sequentially numbered) and values are these item descriptions.
3.  **Generate Python Expression Template:** Write a single-line Python mathematical expression using **only** placeholder keys (`VAL_1`, `VAL_2`, etc.) and standard operators (`+`, `-`, `*`, `/`, `**`, `()`). 
    *   This expression should compute the **raw numerical value**.
    *   Do NOT include actual numbers or dictionary access.
    *   Handle internal scaling ONLY if necessary (e.g., `VAL_1 * 1000 + VAL_2` if VAL_1 is billions, VAL_2 millions).
4.  **Identify Output Format:** Specify if the final answer should be formatted as a 'number' or a 'percent'. 
    *   Use 'percent' if the calculation type is `percentage_change` or `ratio`.
    *   Use 'percent' if the question asks for a comparison between percentages (e.g., "difference in percentage change", "outperform by percent").
    *   **IMPORTANT:** If the calculation type is `average`, `sum`, or `difference`, AND the *input values* being operated on (based on their descriptions in `required_items_map`) are already percentages (e.g., "Effective Tax Rate %"), then the `output_format` MUST be 'number', as the result is already in percentage units.
    *   Otherwise, use 'number'.

**Input Format:**
```text
Question:
[The user's question]
---
Relevant Evidence:
[Formatted evidence chunks]
```

**Output Requirements:**

*   **Format:** Respond using the `{COMBINED_STEP_FUNCTION_NAME}` function call.
*   **Arguments:** A JSON object with `calculation_type`, `required_items_map`, `python_expression_template`, and optionally `output_format` (defaults to 'number').
    *   `required_items_map`: Keys must be `VAL_1`, `VAL_2`, ...
    *   `python_expression_template`: String using *only* placeholder keys and math operators, calculating the raw value.
    *   `output_format`: Must be 'number' or 'percent'.
*   **Strictness:** Output ONLY the function call JSON.

**Example 1 (Percentage Change):**
*Question:* Percentage change in Net Sales from 2000 to 2001?
*Output Args:*
```json
{{
  "calculation_type": "percentage_change",
  "required_items_map": {{ "VAL_1": "Net Sales 2001", "VAL_2": "Net Sales 2000" }},
  "python_expression_template": "(VAL_1 - VAL_2) / VAL_2",
  "output_format": "percent"
}}
```

**Example 2 (Average of Percentages):**
*Question:* Average effective tax rate for 2015 and 2014?
*Output Args:*
```json
{{
  "calculation_type": "average",
  "required_items_map": {{ "VAL_1": "Effective Tax Rate % 2015", "VAL_2": "Effective Tax Rate % 2014" }},
  "python_expression_template": "(VAL_1 + VAL_2) / 2",
  "output_format": "number" # Inputs are percentages, so output is already a percentage number
}}
```

**Example 3 (Difference of Ratios):**
*Question:* What was the difference in percentage change between Index A and Index B from 2010 to 2015?
*Output Args:*
```json
{{
  "calculation_type": "difference", 
  "required_items_map": {{ "VAL_1": "Index A 2015", "VAL_2": "Index A 2010", "VAL_3": "Index B 2015", "VAL_4": "Index B 2010" }},
  "python_expression_template": "((VAL_1 - VAL_2) / VAL_2) - ((VAL_3 - VAL_4) / VAL_4)",
  "output_format": "percent" # Output is the difference between two ratios, format as percent
}}
```
"""
    # Construct messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question:\\n{question}\\n---\\nRelevant Evidence:\\n{evidence_text}"}
    ]

    # API Call
    try:
        # Using default model (gpt-4o-mini unless overridden)
        model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini") 
        logger.info(f"Using OpenAI model: {model_name} for combined specification and expression generation.")
    response = openai.ChatCompletion.create(
            model=model_name,
        messages=messages,
        temperature=0,
            max_tokens=700, 
            functions=[COMBINED_STEP_FUNCTION_SCHEMA],
            function_call={"name": COMBINED_STEP_FUNCTION_NAME}
    )

    message = response.choices[0].message
        if message.get("function_call"):
            func_call = message["function_call"]
            if func_call.get("name") == COMBINED_STEP_FUNCTION_NAME:
                try:
                    args_str = func_call.get("arguments", "{}")
                    args = json.loads(args_str)
                    # --- Basic Validation --- 
                    if not all(k in args for k in ["calculation_type", "required_items_map", "python_expression_template"]):
                         logger.error(f"LLM missing required fields in combined step: {args_str}")
                         return None
                    if not isinstance(args["required_items_map"], dict) or not args["required_items_map"]:
                        logger.error(f"LLM returned invalid/empty required_items_map: {args_str}")
                        return None
                        
                    # --- Simpler Safety Check --- 
                    # Remove complex regex. Check for specific dangerous keywords instead.
                    template_to_check = args["python_expression_template"]
                    disallowed_keywords = ["import", "exec", "eval", "__"]
                    if any(keyword in template_to_check.lower() for keyword in disallowed_keywords):
                        logger.error(f"LLM generated potentially unsafe expression template (contains disallowed keyword): {template_to_check}")
                        return None
                    # --- End Simpler Safety Check --- 

                    # Check placeholder keys are VAL_d+
                    for key in args["required_items_map"].keys():
                        # Replace unreliable regex check with direct string checks
                        is_valid_key = (
                            isinstance(key, str) and 
                            key.startswith("VAL_") and 
                            len(key) > 4 and # Must have at least one digit after VAL_
                            key[4:].isdigit()
                        )
                        if not is_valid_key:
                             # Keep enhanced logging format
                             logger.error(f"LLM used invalid placeholder key format. Expected VAL_ followed by digits. Key checked: >>>{key}<<< (Type: {type(key)}). Full map: {args_str}") 
                             return None
                           
                    args.setdefault("output_format", "number") # Set default for output format
                    logger.info(f"Successfully specified requirements and expression template: {args}")
                    return args 
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode arguments JSON for combined step: {args_str}")
                    return None
            else:
                logger.error(f"LLM called unexpected function in combined step: {func_call.get('name')}")
                return None
        else:
            logger.warning(f"LLM did not generate function call for combined step. Response: {message.get('content')}")
            return None
            
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API call failed (combined step): {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM call (combined step): {e}", exc_info=True)
        return None

# --- Step 2: Extract All Values ---

MULTI_VALUE_EXTRACTION_FUNCTION_NAME = "report_extracted_values"
MULTI_VALUE_EXTRACTION_BASE_SCHEMA = {
    "name": MULTI_VALUE_EXTRACTION_FUNCTION_NAME,
    "description": "Reports the fully scaled numerical values found in the evidence for multiple specified items.",
    "parameters": {
        "type": "object",
        "properties": {}, # Populated dynamically
        # "required": [] # Populated dynamically
    }
}

def extract_all_values_with_llm(
    required_items: List[str],
    evidence_chunks: List[Dict[str, str]]
) -> Tuple[Dict[str, float | int | None], List[str]]: # Return value is now number or None
    """
    Uses a single LLM function call to extract ALL required numerical values, applying unit scaling.
    Returns fully scaled numbers (e.g., millions applied, percentages as decimals) or None.
    """
    logger.info(f"Attempting LLM multi-extraction with scaling for: {required_items}")
    found_values_scaled = {} # Store scaled numeric values or None
    errors = []

    if not required_items:
        logger.warning("extract_all_values_with_llm called with no required items.")
        return {}, []

    # --- Prepare Evidence (Top N Chunks) ---
    MAX_CHUNKS_FOR_PROMPT = 7
    top_evidence = evidence_chunks[:MAX_CHUNKS_FOR_PROMPT]
    if not top_evidence:
        logger.error(f"No evidence chunks provided for multi-extraction.")
        return {}, [f"No evidence provided to find '{item}'" for item in required_items]

    # Pre-process evidence text (same as before)
    processed_chunks_for_prompt = []
    for i, chunk in enumerate(top_evidence):
        chunk_id = chunk.get("chunk_id", f"unknown_{i}")
        text = chunk.get("text", "")
        prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\\n{text}\\n---"
        is_table_chunk = chunk_id.endswith("::table:full") and text.strip().startswith("Financial Table Data:")
        if is_table_chunk:
            md_content = text.split("\\n\\n", 1)[-1]
            parsed_table = parse_markdown_table(md_content)
            if parsed_table:
                try:
                    json_table_string = json.dumps(parsed_table, separators=(',', ':'))
                    prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\\n```json\\n{json_table_string}\\n```\\n---"
                except Exception as json_err:
                    logger.error(f"Failed JSON dump for {chunk_id}: {json_err}")
                    prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\\n{text}\\n---"
            else:
                 prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\\n{text}\\n---"
        processed_chunks_for_prompt.append(prompt_text)
    evidence_text_for_prompt = "\\n".join(processed_chunks_for_prompt)

    # --- Dynamically Create Function Schema (Updated for number/null) ---
    current_schema = MULTI_VALUE_EXTRACTION_BASE_SCHEMA.copy()
    current_schema["parameters"] = current_schema["parameters"].copy()
    current_schema["parameters"]["properties"] = {}
    for item in required_items:
        prop_key = item
        current_schema["parameters"]["properties"][prop_key] = {
            "type": ["number", "null"], # Expect scaled number or null
            "description": f"The fully scaled numerical value for '{item}'. Apply units (millions, %, etc.) found in context. Use null if not found."
        }
    # Requirement implicitly handled by requesting null for missing values
    # current_schema["parameters"]["required"] = list(current_schema["parameters"]["properties"].keys()) # Optional: can keep if strictness needed

    # --- Define System and User Prompts (Updated for Scaling) ---
    numbered_items_list = "\\n".join([f"{i+1}. {item}" for i, item in enumerate(required_items)])

    system_prompt = f"""\
You are a highly accurate data extraction bot. Your task is to find the specific numerical values corresponding to ALL the requested financial items listed below, apply any relevant unit scaling (millions, thousands, billions, percentages), and return the final numeric value using ONLY the provided text and table evidence.

**Your Task:**

1.  Carefully review the entire `List of Items to Extract`.
2.  Scan the `Relevant Evidence` provided (text and tables).
3.  For **EACH item** in the list:
    a.  Locate the **single exact numerical value** within the `Relevant Evidence` that matches the item description (metric name and year/period).
    b.  **CRITICAL: Double-check the year!** Ensure the number corresponds precisely to the year specified.
    c.  **Identify and Apply Units/Scaling:**
        *   Look for units mentioned near the number, in table headers/footnotes, or surrounding text (e.g., "in millions", "thousands", "billions", "%").
        *   **Apply the scaling:**
            *   "millions": Multiply the found number by 1,000,000.
            *   "thousands": Multiply by 1,000.
            *   "billions": Multiply by 1,000,000,000.
            *   "%": Divide the found number by 100 (e.g., 75% becomes 0.75).
        *   Handle negative numbers indicated by parentheses, e.g., (500) becomes -500 *before* scaling.
        *   Remove commas before scaling (e.g., "1,234 million" -> 1234 * 1,000,000 = 1,234,000,000).
    d.  **NEGATIVE CONSTRAINT:** DO NOT extract values mentioned *only* in descriptions of change or difference (e.g., if text says "revenue increased by $302 million", do NOT extract 302,000,000 if you are looking for "Revenue 2017". Find the actual reported value for "Revenue 2017" and scale it).
    e.  Return the final, fully scaled numeric value.

**Examples of Correct Scaling & Formatting:**
*   Text says "revenue was $1.3 million": Return `1300000`
*   Text says "growth of 75%": Return `0.75`
*   Text says "loss of (500) thousands": Return `-500000`
*   Table header says "(in millions)" and cell has `1,234.5`: Return `1234500000`
*   Text says "tax rate 25%": Return `0.25`

**Output Requirements:**

*   **Format:** You MUST respond using the `{MULTI_VALUE_EXTRACTION_FUNCTION_NAME}` function call.
*   **Arguments:** The function call arguments MUST be a JSON object containing **a key for EVERY item** listed in the `List of Items to Extract`.
*   **Values:**
    *   For each key (item description), the value should be the fully scaled **numeric value** (number type in JSON) after applying units/scaling as described above.
    *   If a specific item's value is explicitly stated as zero **in the provided evidence**, use the number `0`.
    *   If you **cannot confidently locate** a specific item's value **within the provided evidence** (even after careful searching, year-checking, and checking for units), use `null` as the value for that item's key.
    *   **You MUST provide a value (`number` or `null`) for every requested item.**
*   **Strictness:** Output ONLY the function call JSON. No explanations.
"""
    # Use the numbered list in the user prompt
    user_prompt = f"""List of Items to Extract:
{numbered_items_list}
---
Relevant Evidence:
```
{evidence_text_for_prompt}
```
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # --- API Call ---
    try:
        # Using default extraction model (gpt-4o-mini unless overridden)
        model_name = os.getenv("OPENAI_CHAT_MODEL_EXTRACTION", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
        logger.info(f"Using OpenAI model: {model_name} for multi-value extraction (with scaling) of {len(required_items)} items.")
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=max(500, 100 + len(required_items) * 50),
            functions=[current_schema],
            function_call={"name": MULTI_VALUE_EXTRACTION_FUNCTION_NAME}
        )

        message = response.choices[0].message
        if message.get("function_call"):
            func_call = message["function_call"]
            if func_call.get("name") == MULTI_VALUE_EXTRACTION_FUNCTION_NAME:
                try:
                    args_str = func_call.get("arguments", "{}")
                    extracted_args = json.loads(args_str)
                    logger.info(f"LLM returned extracted & scaled arguments: {extracted_args}")

                    # Store scaled results (number or None)
                    for item in required_items:
                        value = extracted_args.get(item) # Should be number or null
                        if value is None:
                            logger.warning(f"Value for '{item}' was null (not found or couldn't be scaled).")
                            found_values_scaled[item] = None
                            errors.append(f"Value not found or invalid for: '{item}'")
                        elif isinstance(value, (int, float)):
                            found_values_scaled[item] = value
                            logger.debug(f"Scaled value found for '{item}': {value}")
                        else:
                             # Should not happen if LLM follows instructions
                             logger.error(f"LLM returned non-numeric/non-null value for '{item}': {value} (Type: {type(value)})")
                             found_values_scaled[item] = None
                             errors.append(f"Invalid type returned for '{item}': {type(value)}")

                except json.JSONDecodeError:
                    logger.error(f"Failed to decode arguments JSON for multi-value extraction: {args_str}")
                    errors.extend([f"LLM returned invalid JSON for '{item}'" for item in required_items])
    else:
                logger.error(f"LLM called wrong function for multi-value extraction: {func_call.get('name')}")
                errors.extend([f"LLM called wrong function for '{item}'" for item in required_items])
        else:
            logger.warning(f"LLM did not return function call for multi-value extraction. Response: {message.get('content')}")
            errors.extend([f"LLM did not return function call for '{item}'" for item in required_items])

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API call failed (multi-value extraction): {e}")
        errors.extend([f"OpenAI API error extracting '{item}'" for item in required_items])
    except Exception as e:
        logger.error(f"Unexpected error during LLM call (multi-value extraction): {e}", exc_info=True)
        errors.extend([f"Unexpected error extracting '{item}'" for item in required_items])

    # Final check: If no values were found at all, report errors for all items
    if not found_values_scaled and not errors:
         errors.extend([f"Extraction failed for '{item}' (no response)" for item in required_items])

    # Return the dictionary containing numbers or None
    return found_values_scaled, errors

# --- Orchestration Function ---

def plan_and_execute(
    question: str,
    evidence_chunks: List[Dict[str, str]],
    chat_history: Optional[List[Tuple[str, str]]] = None, # Chat history kept for potential future use
) -> Dict[str, Any]:
    """Orchestrates the agent process: specify_and_express -> extract_all -> substitute_and_eval -> format."""

    # --- Log Received Evidence ---
    if evidence_chunks:
        logger.info(f"--- Evidence Received by Agent (Top {len(evidence_chunks)}) ---")
        for i, chunk in enumerate(evidence_chunks):
            chunk_id = chunk.get("chunk_id", f"unknown_{i}")
            text_preview = chunk.get("text", "").replace("\\n", " ")[:150]
            score = chunk.get("score", None)
            score_str = f" (Score: {score:.4f})" if score is not None else ""
            logger.info(f"[{i+1}] ID: {chunk_id}{score_str} | Text: {text_preview}...")
        logger.info("-------------------------------------------------")
    else:
        logger.warning("Agent received no evidence chunks.")

    # --- Step 1: Specify Requirements and Generate Expression Template (Combined) ---
    logger.info(f"Step 1: Specifying requirements and generating expression template for question: '{question[:50]}...'")
    combined_spec = specify_and_generate_expression(question, evidence_chunks)

    if not combined_spec:
        logger.error("Failed to get valid specification and expression template from LLM.")
        return {
            "answer": "Error: Could not determine calculation requirements or expression.",
            "program": "specify_or_express_failed",
            "intermediates": [],
            "tool": "none",
            "evidence": [chunk.get("chunk_id", "unknown") for chunk in evidence_chunks] if evidence_chunks else [],
        }

    calculation_type = combined_spec.get('calculation_type', 'other')
    required_items_map = combined_spec.get('required_items_map', {})
    python_expression_template = combined_spec.get('python_expression_template')
    output_format = combined_spec.get('output_format', 'number')

    required_item_descriptions = list(required_items_map.values())
    if not required_item_descriptions:
         logger.error(f"No required items identified in map: {required_items_map}")
         return {
             "answer": "Error: No required items identified for calculation.",
             "program": "specify_or_express_failed",
             "intermediates": [combined_spec],
             "tool": "none",
             "evidence": [chunk.get("chunk_id", "unknown") for chunk in evidence_chunks] if evidence_chunks else [],
         }

    logger.info(f"Step 1 successful. Type: {calculation_type}, Format: {output_format}, Template: {python_expression_template}, Required Items: {required_item_descriptions}")

    # --- Step 2: Extract ALL Values using SINGLE LLM Call (with SCALING) ---
    logger.info(f"Step 2: Extracting and scaling {len(required_item_descriptions)} required values via single LLM call...")
    # found_values_raw changed to found_values_scaled
    found_values_scaled, extraction_errors = extract_all_values_with_llm(required_item_descriptions, evidence_chunks)

    if extraction_errors:
        logger.warning(f"Extraction step reported errors (may be partial): {extraction_errors}")

    logger.info(f"Step 2 finished. Scaled extracted values: {found_values_scaled}")

    # --- Step 3: Substitute Values into Expression Template ---
    logger.info("Step 3: Substituting extracted & scaled values into expression template...")
    placeholder_to_value_map = {} # Map placeholder (e.g., VAL_1) to scaled numeric value or None
    substitution_errors = []

    # Sort placeholders (VAL_1, VAL_2...) for consistent processing
    sorted_placeholders = sorted(required_items_map.keys(), key=lambda x: int(x.split('_')[1]))

    for placeholder in sorted_placeholders:
        description = required_items_map.get(placeholder)
        if not description:
             logger.error(f"Internal Error: Placeholder {placeholder} has no description in map.")
             substitution_errors.append(f"Internal error mapping {placeholder}")
             continue

        # Use the scaled value directly (should be number or None)
        scaled_value = found_values_scaled.get(description)

        # Check if value is missing (None)
        if scaled_value is None:
            logger.warning(f"Value for item '{description}' (placeholder {placeholder}) was not extracted/scaled or is null.")
            substitution_errors.append(f"Missing value for {placeholder} ('{description}')")
            # Don't add to map, fail later if template requires it
            continue
        elif isinstance(scaled_value, (int, float)):
             placeholder_to_value_map[placeholder] = scaled_value
             logger.debug(f"Mapped {placeholder} ('{description}') to scaled value {scaled_value}")
        else:
            # Should not happen if extract_all_values_with_llm works correctly
            logger.error(f"Unexpected non-numeric type for scaled value of '{description}' ({placeholder}): {type(scaled_value)}")
            substitution_errors.append(f"Invalid scaled value type for {placeholder}")
            continue

    # --- Fail Fast if Required Values are Missing for the Template ---
    placeholders_in_template = set(re.findall(r'VAL_\d+', python_expression_template))
    missing_placeholders_in_template = []
    for ph in placeholders_in_template:
        if ph not in placeholder_to_value_map:
            missing_desc = required_items_map.get(ph, "Unknown Item")
            missing_placeholders_in_template.append(f"{ph} ('{missing_desc}')")

    if missing_placeholders_in_template:
         all_errors = substitution_errors + [f"Missing required value for {ph_desc}" for ph_desc in missing_placeholders_in_template]
         error_message = "; ".join(sorted(list(set(all_errors))))
         logger.error(f"Cannot substitute values for template '{python_expression_template}' due to missing required values: {error_message}")
         return {
             "answer": f"Error preparing calculation: {error_message}",
             "program": "substitute_failed",
             "intermediates": [combined_spec, found_values_scaled], # Show scaled extracted values
             "tool": "none",
             "evidence": [chunk.get("chunk_id", "unknown") for chunk in evidence_chunks] if evidence_chunks else [],
         }
    # --- End Fail Fast ---

    # Perform the substitution
    final_expression_string = python_expression_template
    try:
        def replace_placeholder(match):
             placeholder = match.group(0)
             # Ensure the value from the map is used directly
             value = placeholder_to_value_map[placeholder]
             return str(value) # Convert number to string for substitution

        final_expression_string = re.sub(r'VAL_\d+', replace_placeholder, final_expression_string)
        logger.info(f"Step 3 successful. Final expression for eval: {final_expression_string}")

    except Exception as sub_err:
         logger.error(f"Unexpected error during value substitution: {sub_err}", exc_info=True)
         return {
             "answer": f"Error during calculation preparation: {sub_err}",
             "program": "substitute_failed",
             "intermediates": [combined_spec, found_values_scaled, {"template": python_expression_template}],
             "tool": "none",
             "evidence": [chunk.get("chunk_id", "unknown") for chunk in evidence_chunks] if evidence_chunks else [],
         }

    # --- Step 4: Execute Python Expression using safe eval() ---
    logger.info(f"Step 4: Executing expression: {final_expression_string}")
    final_answer_val = None
    error_message = ""

    safe_globals = {"__builtins__": None, 'abs': abs, 'pow': pow, 'round': round}
    safe_locals = {}

    try:
        final_answer_val = eval(final_expression_string, safe_globals, safe_locals)

        if not isinstance(final_answer_val, (int, float)):
             logger.warning(f"Eval result was not a number: {type(final_answer_val)} - {final_answer_val}")
             try:
                 final_answer_val = float(final_answer_val)
             except (ValueError, TypeError):
                 error_message = f"Error: Calculation result '{final_answer_val}' is not a valid number."
                 final_answer_val = None

    except SyntaxError as e:
        error_message = f"Error: Invalid calculation syntax ({e}) in '{final_expression_string}'."
        logger.error(error_message)
    except NameError as e:
        error_message = f"Error: Calculation tried to use an undefined variable ({e}) in '{final_expression_string}'."
        logger.error(error_message)
    except TypeError as e:
        error_message = f"Error: Type mismatch during calculation ({e}) in '{final_expression_string}'."
        logger.error(error_message)
    except ZeroDivisionError:
        error_message = f"Error: Division by zero during calculation in '{final_expression_string}'."
        logger.error(error_message)
    except Exception as e:
        error_message = f"Error: Unexpected error during calculation execution ({e}) in '{final_expression_string}'."
        logger.error(error_message, exc_info=True)

    if final_answer_val is None:
        return {
            "answer": error_message,
            "program": "eval_failed",
            "intermediates": [combined_spec, found_values_scaled, {"substituted_expression": final_expression_string}],
            "tool": "python_eval",
            "evidence": [chunk.get("chunk_id", "unknown") for chunk in evidence_chunks] if evidence_chunks else [],
        }

    # --- Step 5: Format Final Answer (Using output_format flag) ---

    # --- START: Output Format Override Check (No longer needed if scaling is done correctly in Step 2) ---
    # Keeping this commented out for now, as Step 2 should handle % -> decimal conversion.
    # inputs_were_percentages = False
    # if calculation_type in ['average', 'sum', 'difference']:
    #     if any('%' in desc for desc in required_item_descriptions): # This check might be less reliable now
    #         inputs_were_percentages = True
    #         logger.info(f"Detected calculation ('{calculation_type}') on percentage inputs. Forcing output format to 'number'. Original: '{output_format}'")
    #         output_format = 'number' # Override!
    # --- END: Output Format Override Check ---

    final_answer_str = ""
    logger.info(f"Formatting result: {final_answer_val} based on final output_format: {output_format}")

    # Formatting depends only on output_format.
    try:
        if output_format == 'percent':
            # Assuming final_answer_val is already a ratio (e.g., 0.086 for 8.6%)
            final_answer_str = f"{final_answer_val:.1%}"
            logger.info(f"Formatted percentage answer: {final_answer_str}")

        elif isinstance(final_answer_val, (int, float)):
             if isinstance(final_answer_val, float):
                  # Check for "integer floats" like 62.0 - format as int
                  if final_answer_val == int(final_answer_val):
                      final_answer_str = str(int(final_answer_val))
                  else:
                      final_answer_str = f"{final_answer_val:.2f}"
             else:
                  final_answer_str = str(final_answer_val)
             logger.info(f"Formatted number answer: {final_answer_str}")
        else:
             logger.warning(f"Unexpected answer type before formatting: {type(final_answer_val)}")
             final_answer_str = str(final_answer_val)

    except Exception as format_err:
        logger.error(f"Error formatting final answer '{final_answer_val}' (output_format: {output_format}): {format_err}", exc_info=True)
        final_answer_str = f"Error formatting result: {final_answer_val}"

    logger.info(f"Step 4 & 5 successful. Final formatted answer: {final_answer_str}")

    # --- Step 6: Return Result ---
    return {
        "answer": final_answer_str,
        "program": python_expression_template, # Store the template
        "intermediates": [combined_spec, found_values_scaled, {"substituted_expression": final_expression_string}],
        "tool": "python_eval",
        "evidence": [chunk.get("chunk_id", "unknown") for chunk in evidence_chunks] if evidence_chunks else [],
    }
