"""
########################################################################
# FinRAG Agent Module (agent.py)
#
# This module implements the "planning and execution" stage of the RAG pipeline:
# 1. It accepts reranked evidence chunks and a user question.
# 2. It formats the evidence into prompts for the LLM's function-calling API.
# 3. generate_tool_call() constructs and validates a DSL program that computes
#    the answer by invoking the math tool.
# 4. plan_and_execute() orchestrates the tool call: it invokes generate_tool_call(),
#    executes the returned DSL via run_math_tool(), captures intermediate values,
#    formats the final answer, and returns a structured dict containing:
#    - program: the DSL string
#    - intermediates: the numeric steps
#    - answer: the final formatted result
#
# Main functions:
# - generate_tool_call(question, evidence_chunks, chat_history=None):
#     Builds the function-calling JSON payload for the LLM based on question & evidence.
# - plan_and_execute(question, evidence_chunks, chat_history=None):
#     Runs generate_tool_call, executes the returned program, handles errors & fallbacks,
#     and composes the final output.
########################################################################
"""

"""Agent module for FinRAG."""

import json
import openai
import warnings
import re
import os
from typing import Any, Dict, List, Tuple, Optional
from finrag.tools import MathToolInput, run_math_tool
import logging

# --- START ADDITION: Moved Number Cleaning Function ---
def clean_numeric_string(s: str) -> str:
    """Clean a string potentially representing a number to a pure numeric format."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip() # Ensure input is string and stripped

    # --- START: Pre-cleaning: Remove common units/qualifiers (case-insensitive) ---
    s_lower = s.lower()
    # Remove currency symbols first (before checking for 'usd', etc.)
    s = s.replace('$', '').replace('£', '').replace('€', '')

    # Words/Units to remove (add more as needed)
    words_to_remove = [
        "million", "billion", "thousand", "trillion",
        "approx", "approximately", "around", "about",
        "usd", "eur", "gbp",
        "years", "year"
        # Be careful adding short words like 'k' - might remove legitimate parts
    ]
    # Remove whole words using regex word boundaries to avoid partial matches
    for word in words_to_remove:
        s = re.sub(r'\b' + re.escape(word) + r'\b', '', s, flags=re.IGNORECASE)

    # After removing words, strip whitespace again
    s = s.strip()
    # --- END: Pre-cleaning ---

    # Handle parentheses for negative numbers: e.g., (500) -> -500
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    
    # Remove remaining currency symbols and commas (redundant $ removal, but safe)
    s = s.replace('$', '').replace(',', '')
    
    # Handle percentage sign: e.g., 15% -> 0.15
    if s.endswith('%'):
        # Ensure the part before % is potentially numeric before converting
        num_part = s[:-1].strip()
        try:
            # Attempt to convert the part before '%' to float and divide by 100
            return str(float(num_part) / 100.0)
        except ValueError:
             # If conversion fails, just return the cleaned numeric part without %
             # This handles cases like '(15)%' -> '-15' (after parenthesis step)
             return num_part
             
    return s
# --- END ADDITION ---

# logger for debugging
logger = logging.getLogger(__name__)

FUNCTION_NAME = "execute_dsl"
FUNCTION_DEFINITIONS = [
    {
        "name": FUNCTION_NAME,
        "description": "Executes a domain-specific language program to answer financial questions based on provided evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "program": {
                    "type": "string",
                    "description": "The DSL program string. Example: 'add(find_metric(\"Net Sales\", \"2022\"), find_metric(\"Other Income\", \"2022\"))'"
                }
            },
            "required": ["program"]
        }
    }
]

# --- START: Add Markdown Table Parser --- 
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
# --- END: Add Markdown Table Parser ---

def is_valid_dsl(program: str) -> bool:
    """Validate DSL program string format."""
    pattern = r'^[a-zA-Z_]+\([^)]*\)(,\s*[a-zA-Z_]+\([^)]*\))*$'
    return bool(re.fullmatch(pattern, program.strip()))

def answer(question: str, conversation_history: List[Dict], doc_id: str) -> Dict:
    """Answer a question given history and document ID."""
    raise NotImplementedError

# --- START: Simplified Function Schema for Requirement Specification ---
SPECIFICATION_FUNCTION_NAME = "specify_calculation_requirements"
SPECIFICATION_FUNCTION_DEFINITION = {
    "name": SPECIFICATION_FUNCTION_NAME,
    "description": "Specifies the calculation type and data points needed.",
    "parameters": {
        "type": "object",
        "properties": {
            "calculation_type": {
                "type": "string",
                "description": "Type of calculation (e.g., 'percentage_change', 'sum', 'average').",
                "enum": ["percentage_change", "sum", "average", "ratio", "difference", "value_lookup", "other"]
            },
            "required_items": {
                "type": "array",
                "description": "List of specific data items needed, like 'Net Sales 2021' or 'Shares Purchased Nov-14'.",
                "items": {
                    "type": "string",
                    "description": "A specific data point (e.g., 'Metric Name YYYY')"
                }
            },
             "unit_requirement": {
                 "type": "string",
                 "description": "Unit conversion needed? ('millions', 'thousands', or 'none').",
                 "enum": ["millions", "thousands", "none"],
                 "default": "none"
             }
        },
        "required": ["calculation_type", "required_items"]
    }
}
# --- END: Simplified Function Schema ---

def generate_tool_call(
    question: str,
    evidence_chunks: List[Dict[str, str]],
    chat_history: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Use LLM function calling to specify calculation requirements based on the question and evidence.
    Rewritten prompt following guidelines.
    """
    # --- Evidence pre-processing remains the same ---
    processed_chunks_for_prompt = []
    for i, chunk in enumerate(evidence_chunks):
        chunk_id = chunk.get("chunk_id", f"unknown_{i}")
        text = chunk.get("text", "")
        prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\n{text}\n---" # Default format
        is_table_chunk = chunk_id.endswith("::table:full") and text.strip().startswith("Financial Table Data:")
        if is_table_chunk:
            md_content = text.split("\n\n", 1)[-1]
            parsed_table = parse_markdown_table(md_content)
            if parsed_table:
                try:
                    json_table_string = json.dumps(parsed_table, separators=(',', ':'))
                    prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\n```json\n{json_table_string}\n```\n---"
                    logger.debug(f"Parsed table chunk {chunk_id} into compact JSON.")
                except Exception as json_err:
                    logger.error(f"Failed JSON dump for {chunk_id}: {json_err}")
            else:
                logger.warning(f"Failed table parse for {chunk_id}.")
        processed_chunks_for_prompt.append(prompt_text)
    evidence_text = "\n".join(processed_chunks_for_prompt)
    # --- End Evidence pre-processing ---

    # --- START: New System Prompt (Requirement Specification) ---
    system_prompt = """\
You are a meticulous financial analyst assistant. Your primary function is to analyze a user's financial question and the provided evidence to determine the precise calculation steps and data points needed to answer it.

**Your Task:**

1.  **Identify Calculation Type:** Determine the core mathematical operation required (e.g., percentage change, sum, average, ratio, difference, direct value lookup). Use 'ratio' when comparing one part to a total or another part (e.g., 'X as a percentage of Y').
2.  **Identify Required Data Items:** List *all* specific numerical data points needed for the calculation. **Format each item concisely as "Metric Name [Optional Context] YYYY/Period"**. Examples:
    *   "Net Sales 2021"
    *   "Operating Income Q3 2020"
    *   "Shares Outstanding Dec 31, 2019"
    *   "Minimum Payment [Capital Leases] 2008" (Use brackets for context if needed for clarity)
    *   "Receivables from Money Pool [Entergy Mississippi] 2014"
    Be precise but avoid overly long descriptions in the item name itself.
3.  **Identify Unit Requirement:** Specify if the *final* answer needs to be scaled to millions or thousands, or if no scaling is needed.

**Input Format:**

You will be given:

```text
Question:
[The user's question]
---
Relevant Evidence:
[Formatted evidence chunks, potentially including JSON tables]
```

**Output Requirements:**

*   **Format:** Your response MUST be a single JSON object representing a function call to `specify_calculation_requirements`.
*   **Content:** The JSON object must contain the required arguments:
    *   `calculation_type`: (String enum: "percentage_change", "sum", "average", "ratio", "difference", "value_lookup", "other")
    *   `required_items`: (List of Strings) - Each string concisely identifies a needed data point following the "Metric [Context] YYYY" format.
    *   `unit_requirement`: (String enum: "millions", "thousands", "none") - Defaults to "none" if not specified.
*   **Strictness:** Do NOT include *any* explanations, apologies, or conversational text. Output ONLY the JSON function call.

**Examples:**

*Example 1 (Percentage Change):*

*Input Question:* What was the percentage change in Net Sales from 2000 to 2001, in millions?
*Input Evidence:* [...]
*Output:*
```json
{
    "function_call": {
        "name": "specify_calculation_requirements",
        "arguments": "{\\"calculation_type\\": \\"percentage_change\\", \\"required_items\\": [\\"Net Sales 2001\\", \\"Net Sales 2000\\"], \\"unit_requirement\\": \\"millions\\"}"
    }
}
```

*Example 2 (Average):*

*Input Question:* What was the average Operating Income from 2020 to 2022?
*Input Evidence:* [...]
*Output:*
```json
{
    "function_call": {
        "name": "specify_calculation_requirements",
        "arguments": "{\\"calculation_type\\": \\"average\\", \\"required_items\\": [\\"Operating Income 2020\\", \\"Operating Income 2021\\", \\"Operating Income 2022\\"], \\"unit_requirement\\": \\"none\\"}"
    }
}
```

*Example 3 (Value Lookup):*

*Input Question:* What were Total Assets in 2015?
*Input Evidence:* [...]
*Output:*
```json
{
    "function_call": {
        "name": "specify_calculation_requirements",
        "arguments": "{\\"calculation_type\\": \\"value_lookup\\", \\"required_items\\": [\\"Total Assets 2015\\"], \\"unit_requirement\\": \\"none\\"}"
    }
}
```

*Example 4 (Ratio with Context):*

*Input Question:* What percentage of total minimum lease payments were capital leases in 2008?
*Input Evidence:* [...]
*Output:*
```json
{
    "function_call": {
        "name": "specify_calculation_requirements",
        "arguments": "{\\"calculation_type\\": \\"ratio\\", \\"required_items\\": [\\"Minimum Payment [Capital Leases] 2008\\", \\"Minimum Payment [Operating Leases] 2008\\"], \\"unit_requirement\\": \\"none\\"}"
    }
}
```

*Example 5 (Difference with Multiple Inputs):*

*Input Question:* How much more was spent on purchased shares in October than in November 2018?
*Input Evidence:* [...]
*Output:*
```json
{
    "function_call": {
        "name": "specify_calculation_requirements",
        "arguments": "{\\"calculation_type\\": \\"difference\\", \\"required_items\\": [\\"Shares Purchased October 2018\\", \\"Avg Price October 2018\\", \\"Shares Purchased November 2018\\", \\"Avg Price November 2018\\"], \\"unit_requirement\\": \\"none\\"}"
    }
}
```
"""
    # --- END: New System Prompt ---

    # Construct messages for the API call
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Add the actual user question and evidence, using delimiters
    user_prompt = f"""Question:
{question}
---
Relevant Evidence:
{evidence_text}
"""
    messages.append({"role": "user", "content": user_prompt})

    # Make the API call
    try:
        model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        logger.info(f"Using OpenAI model: {model_name} for specification")

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=400,
            functions=[SPECIFICATION_FUNCTION_DEFINITION], # Use schema defined above
            function_call={"name": SPECIFICATION_FUNCTION_NAME} # Force call
        )

        message = response.choices[0].message
        if message.get("function_call"):
            func_call = message["function_call"]
            if func_call.get("name") == SPECIFICATION_FUNCTION_NAME:
                try:
                    args_str = func_call.get("arguments", "{}")
                    args = json.loads(args_str)
                    # Validate required fields are present and required_items is a list
                    if ("calculation_type" in args and
                            "required_items" in args and
                            isinstance(args["required_items"], list)):
                        args.setdefault("unit_requirement", "none") # Ensure default
                        logger.info(f"Successfully specified requirements: {args}")
                        return args # Return the parsed arguments dictionary
                    else:
                        logger.error(f"LLM returned invalid structure for requirements: {args_str}")
                        return {}
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode arguments JSON for requirements: {args_str}")
                    return {}
            else:
                logger.error(f"LLM called unexpected function: {func_call.get('name')}")
                return {}
        else:
            logger.warning(f"LLM did not generate a function call for specification. Response: {message.get('content')}")
            return {}

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API call failed (specification): {e}")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM call (specification): {e}")
        return {}

def plan_and_execute(
    question: str,
    evidence_chunks: List[Dict[str, str]],
    chat_history: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Orchestrates the agent process: specify -> extract (LLM per item) -> generate -> execute."""
    
    # --- Step 1: Specify Requirements using LLM (No Change) --- 
    logger.info(f"Step 1: Specifying requirements for question: '{question[:50]}...'")
    requirements = generate_tool_call(question, evidence_chunks, chat_history)
    
    required_items_list = requirements.get('required_items', [])
    if not requirements or not required_items_list: 
        logger.error("Failed to get valid requirements (missing required_items) from LLM.")
        return {
            "answer": "Error: Could not determine calculation requirements.",
            "program": "", "intermediates": [], "tool": "none", 
            "evidence": [chunk.get("chunk_id", "unknown") for chunk in evidence_chunks],
        }
    logger.info(f"Step 1 successful. Requirements: {requirements}")

    # --- Step 2: Extract Values using LLM (One call per required item) --- 
    logger.info(f"Step 2: Extracting {len(required_items_list)} required values via LLM...")
    found_values = {} 
    extraction_errors = []
    
    for item_string in required_items_list:
        extracted_value = extract_single_value_with_llm(item_string, evidence_chunks)
        if extracted_value is not None:
            found_values[item_string] = extracted_value
        else:
            # Log implicitly done in extract_single_value_with_llm if value is None
            extraction_errors.append(f"Failed to extract value for: '{item_string}'") 
            
    # Check if ANY extraction failed
    if extraction_errors:
        error_message = "; ".join(extraction_errors)
        logger.error(f"Errors during LLM extraction: {error_message}")
        # Decide if we should proceed with partial data or fail completely.
        # For now, failing completely if any value is missing seems safer for calculations.
        return {
            "answer": f"Error during data extraction: {error_message}",
            "program": "", "intermediates": [], "tool": "none", 
            "evidence": [chunk.get("chunk_id", "unknown") for chunk in evidence_chunks],
        }
        
    # Optional: Double-check if all originally required keys are present 
    # (Should be guaranteed by the check above if we fail on any error)
    # required_keys_set = set(required_items_list)
    # if not required_keys_set.issubset(found_values.keys()):
    #     missing_keys = required_keys_set - found_values.keys()
    #     error_message = f"Internal Error: Missing required values after extraction despite no errors: {missing_keys}"
    #     logger.error(error_message)
    #     return {
    #         "answer": f"Error: {error_message}",
    #         "program": "", "intermediates": [], "tool": "none",
    #         "evidence": [chunk.get("chunk_id", "unknown") for chunk in evidence_chunks],
    #     }

    logger.info(f"Step 2 successful. Found values: {found_values}")

    # --- Step 3: Generate DSL Program using LLM (No Change in Call Signature) --- 
    logger.info("Step 3: Generating DSL program...")
    generated_program = generate_dsl_program_from_values(
        calculation_type=requirements.get('calculation_type', 'other'),
        required_items=required_items_list, # Pass the original list 
        found_values=found_values, # Pass the dict of successfully found values
        unit_requirement=requirements.get('unit_requirement', 'none')
    )
    
    if not generated_program:
        logger.error("Failed to generate DSL program.")
        # Include found values in the error message for debugging
        return {
            "answer": f"Error: Failed to generate calculation program from extracted values: {found_values}",
            "program": "",
            "intermediates": [],
            "tool": "none", 
            "evidence": [chunk.get("chunk_id", "unknown") for chunk in evidence_chunks],
        }
    logger.info(f"Step 3 successful. Generated program: {generated_program}")

    # --- Step 4: Execute DSL Program (No Change) --- 
    logger.info(f"Step 4: Executing DSL program: {generated_program}")
    tool_name = FUNCTION_NAME # Use the original tool name expected by evaluation
    try:
        math_input = MathToolInput(program=generated_program)
        result = run_math_tool(math_input)
        final_answer = result.get("answer", "Error: Execution produced no answer")
        intermediates = result.get("intermediates", [])
        logger.info(f"Step 4 successful. Execution result: {final_answer}")
    except (ValueError, IndexError, TypeError, ZeroDivisionError) as e:
        logger.error(f"Error executing generated DSL program '{generated_program}': {e}")
        final_answer = f"Error executing '{generated_program}': {e}"
        intermediates = []
    except Exception as e:
        logger.error(f"Unexpected error executing generated DSL program '{generated_program}': {e}")
        final_answer = f"Error: Unexpected execution error: {e}"
        intermediates = []
    
    return {
        "answer": final_answer, 
        "program": generated_program, 
        "intermediates": intermediates,
        "tool": tool_name, 
        "evidence": [chunk.get("chunk_id", "unknown") for chunk in evidence_chunks],
    }

# --- Need to update extraction logic to handle new 'required_items' format --- 

def extract_required_values(requirements: Dict[str, Any], evidence_chunks: List[Dict[str, str]]) -> Tuple[Dict[str, float | int], List[str]]:
    """
    Extracts the numerical values specified in 'requirements' ('required_items') from the 'evidence_chunks'.
    Includes improved JSON search logic AND text fallback search.
    """
    required_items_list = requirements.get('required_items', [])
    found_values = {}
    errors = []
    processed_evidence = []

    # Pre-process evidence: Parse JSON strings back into objects for easier lookup
    for i, chunk in enumerate(evidence_chunks):
        chunk_id = chunk.get("chunk_id", f"unknown_{i}")
        text = chunk.get("text", "")
        parsed_json = None
        is_json_like = text.strip().startswith('[') and text.strip().endswith(']')
        if is_json_like:
            try:
                parsed_json = json.loads(text)
                if not isinstance(parsed_json, list):
                    parsed_json = None
                    logger.warning(f"Evidence chunk {chunk_id} looked like JSON but parsed to non-list type.")
            except json.JSONDecodeError:
                parsed_json = None
                logger.warning(f"Failed to parse potential JSON in evidence chunk {chunk_id}")
        processed_evidence.append({"id": chunk_id, "text": text, "json_content": parsed_json})

    # Iterate through each required item string (e.g., "Net Sales 2021")
    for item_string in required_items_list:
        parts = item_string.split()
        if len(parts) < 2:
            errors.append(f"Could not parse required item string: '{item_string}'")
            continue
        period = parts[-1]
        metric = " ".join(parts[:-1])
        value_key = item_string
        
        found = False
        extracted_value_str = None # Keep track of raw value for logging

        # --- 1. Search JSON tables --- 
        for evidence in processed_evidence:
            if evidence["json_content"]:
                # Metric as Value, Period as Key
                metric_match_key = None
                for key, cell_value in evidence["json_content"][0].items():
                    if isinstance(cell_value, str) and metric.strip().lower() == cell_value.strip().lower():
                        metric_match_key = key
                        break
                if metric_match_key and period in evidence["json_content"][0]:
                    raw_value = evidence["json_content"][0][period]
                    try:
                        cleaned_value_str = clean_numeric_string(raw_value)
                        numeric_value = float(cleaned_value_str) if '.' in cleaned_value_str else int(cleaned_value_str)
                        found_values[value_key] = numeric_value
                        extracted_value_str = raw_value
                        logger.info(f"Found JSON value (M:Val,P:Key) for '{item_string}': {numeric_value} from {evidence['id']}")
                        found = True
                        break
                    except (ValueError, TypeError):
                        logger.warning(f"JSON value conversion failed for '{item_string}' (Raw: {raw_value}) in {evidence['id']}. Trying next.")

                # Period as Value, Metric as Key
                if not found:
                    period_match_key = None
                    for key, cell_value in evidence["json_content"][0].items():
                        if isinstance(cell_value, str) and period.strip().lower() == cell_value.strip().lower():
                            period_match_key = key
                            break
                    if period_match_key and metric in evidence["json_content"][0]:
                        raw_value = evidence["json_content"][0][metric]
                        try:
                            cleaned_value_str = clean_numeric_string(raw_value)
                            numeric_value = float(cleaned_value_str) if '.' in cleaned_value_str else int(cleaned_value_str)
                            found_values[value_key] = numeric_value
                            extracted_value_str = raw_value
                            logger.info(f"Found JSON value (P:Val,M:Key) for '{item_string}': {numeric_value} from {evidence['id']}")
                            found = True
                            break
                        except (ValueError, TypeError):
                            logger.warning(f"JSON value conversion failed for '{item_string}' (Raw: {raw_value}) in {evidence['id']}. Trying next.")
            if found: break 
        # --- END JSON Search ---

        # --- 2. Fallback: Search Raw Text --- 
        if not found:
            logger.info(f"Value for '{item_string}' not found in JSON, attempting text search.")
            # Simple Regex to find numbers near metric and period
            # Look for patterns like "Metric... Period... Value", "Value ... Metric ... Period", etc.
            # This is basic and might need refinement based on common text structures.
            metric_pattern = re.escape(metric.strip())
            period_pattern = re.escape(period.strip())
            # Regex to find number-like strings (supports integers, decimals, negatives, commas, parenthesis)
            number_pattern = r'[-+]?\s*\(?\s*\$?[\d,]+\.?\d*\s*\)?%?' 
            
            best_match_value = None
            best_match_evidence_id = None
            min_dist = float('inf')

            for evidence in processed_evidence:
                text_lower = evidence['text'].lower() # Search case-insensitively
                metric_indices = [m.start() for m in re.finditer(metric_pattern.lower(), text_lower)]
                period_indices = [m.start() for m in re.finditer(period_pattern.lower(), text_lower)]
                number_matches = list(re.finditer(number_pattern, evidence['text'])) # Use original case for extraction

                if not metric_indices or not period_indices or not number_matches:
                    continue # Need metric, period, and numbers in the chunk

                # Find the closest number to *both* a metric and a period occurrence
                for num_match in number_matches:
                    num_start, num_end = num_match.span()
                    num_center = (num_start + num_end) / 2
                    raw_num_str = num_match.group(0)

                    # Find min distance from this number to *any* metric and *any* period
                    dist_to_metric = min(abs(num_center - m_idx) for m_idx in metric_indices)
                    dist_to_period = min(abs(num_center - p_idx) for p_idx in period_indices)
                    
                    # Heuristic: score based on proximity to both (sum of distances)
                    current_dist = dist_to_metric + dist_to_period

                    # Prioritize numbers that are reasonably close to both
                    # Avoid cases where metric and period are far apart but a number is between them
                    max_metric_period_dist = max(abs(m - p) for m in metric_indices for p in period_indices) 
                    
                    # Adjust heuristic: Consider distance relative to metric-period distance? 
                    # Simpler for now: just find the closest overall number to both.
                    
                    if current_dist < min_dist:
                        # Basic check: Is the number plausible?
                        try:
                            _ = clean_numeric_string(raw_num_str) # Test cleaning
                            min_dist = current_dist
                            best_match_value = raw_num_str
                            best_match_evidence_id = evidence['id']
                        except ValueError:
                            pass # Ignore if cleaning fails

            if best_match_value:
                try:
                    cleaned_value_str = clean_numeric_string(best_match_value)
                    numeric_value = float(cleaned_value_str) if '.' in cleaned_value_str else int(cleaned_value_str)
                    found_values[value_key] = numeric_value
                    extracted_value_str = best_match_value # Use the raw extracted string
                    logger.info(f"Found TEXT value for '{item_string}': {numeric_value} (Raw: '{extracted_value_str}') in {best_match_evidence_id} (dist: {min_dist:.1f})")
                    found = True
                except ValueError:
                    logger.warning(f"Text value conversion failed for '{item_string}' (Raw: '{best_match_value}') in {best_match_evidence_id}")
                    # errors.append(f"Could not convert text value '{best_match_value}' for '{item_string}' in {best_match_evidence_id}")
            if not found:
                logger.error(f"Required value NOT FOUND in JSON or Text: '{item_string}'") # Updated error
                errors.append(f"Required value not found in evidence: '{item_string}'")

    return found_values, errors

# --- Need to update DSL generation logic to handle new 'required_items' format ---

# Define the function schema for the DSL generation LLM call *outside* the function
# to avoid redefining it on every call.
DSL_GENERATION_FUNCTION_NAME = "generate_dsl"
DSL_GENERATION_FUNCTION_DEFINITION = {
    "name": DSL_GENERATION_FUNCTION_NAME,
    "description": "Generates the final DSL program string based on calculation type and values.",
    "parameters": {
        "type": "object",
        "properties": {
            "program": {
                "type": "string",
                "description": "The DSL program string. Example: 'subtract(5363, 7983), divide(#0, 7983)'"
            }
        },
        "required": ["program"]
    }
}

def generate_dsl_program_from_values(
    calculation_type: str,
    required_items: List[str], # List of strings like "Metric Year"
    found_values: Dict[str, float | int], # Dict mapping "Metric Year" -> value
    unit_requirement: str
) -> str:
    """Generates the DSL program string using an LLM, given the calculation type and extracted values."""
    
    value_map_for_prompt = found_values # Pass the dict directly

    # Check if all required items were actually found
    missing_keys = [item for item in required_items if item not in found_values]
    if missing_keys:
        logger.error(f"DSL Generation called but missing required values: {missing_keys}. Cannot generate program.")
        return ""

    # --- START: New System Prompt (DSL Generation) ---
    system_prompt = f"""\
You are a specialized DSL (Domain-Specific Language) code generator. Your task is to translate abstract calculation requirements and concrete numerical values into a precise DSL program string.

**DSL Specification:**

*   **Available Functions:** You can ONLY use the following functions: `add`, `subtract`, `multiply`, `divide`.
*   **Arguments:** Functions take exactly two arguments, which can be numbers or references to intermediate results.
*   **Intermediate Results:** Use `#N` to refer to the result of the Nth previous operation (0-indexed).
*   **Values:** Use the *exact* numerical values provided in the `Available Numerical Values` input.
*   **Unit Conversion (IMPORTANT JUDGMENT REQUIRED):**
    *   Examine the `Unit Requirement` AND the magnitude of the numbers in `Available Numerical Values`.
    *   Apply scaling ONLY if the numbers appear **unscaled** relative to the requirement. 
    *   Example 1: If `Unit Requirement` is 'millions' and a key number is `123456789`, it likely needs scaling. Add `, divide(#N, 1000000)` as the *final* step.
    *   Example 2: If `Unit Requirement` is 'millions' but a key number is `123.45`, it likely **already represents millions**. Do **NOT** add the scaling step.
    *   Example 3: If `Unit Requirement` is 'billions' and a key number is `3.5`, it likely **already represents billions**. Do **NOT** add scaling.
    *   If `Unit Requirement` is 'none', never add scaling.
*   **Structure:** The final output MUST be a comma-separated sequence of **binary** function calls. **Do NOT nest function calls** (e.g., use `add(1, 2), add(#0, 3)` instead of `add(add(1, 2), 3)`).

**Input Format:**

You will receive the calculation details structured as follows:

```
Calculation Type: {calculation_type}
---
Needed Items (Keys for Values):
{json.dumps(required_items, indent=2)}
---
Available Numerical Values:
{json.dumps(value_map_for_prompt, indent=2)}
---
Unit Requirement: {unit_requirement}
```

**Output Requirements:**

*   **Format:** Your response MUST be a single JSON object representing a function call to `generate_dsl`.
*   **Content:** The JSON object must contain the `program` argument, holding the generated DSL string, adhering strictly to the sequential, binary structure.
*   **Strictness:** Output ONLY the JSON function call. No extra text.

**Example (Corrected Scenario):**

*Input:*
```
Calculation Type: percentage_change
---
Needed Items (Keys for Values):
[
  "Asset Allocation Dec 31, 2015",
  "Asset Allocation Dec 31, 2014"
]
---
Available Numerical Values:
{{
  "Asset Allocation Dec 31, 2015": 185836,
  "Asset Allocation Dec 31, 2014": 183032
}}
---
Unit Requirement: millions 
```

*Output (No Scaling Applied, Sequential Structure):*
```json
{{
    "function_call": {{
        "name": "generate_dsl",
        "arguments": "{{\\"program\\": \\"subtract(185836, 183032), divide(#0, 183032)\\"}}"
    }}
}}
```

**Example (Sequential Addition):**

*Input:*
```
Calculation Type: average
---
Needed Items (Keys for Values): ["Val A", "Val B", "Val C"]
---
Available Numerical Values: {{"Val A": 10, "Val B": 20, "Val C": 30}}
---
Unit Requirement: none
```

*Output:*
```json
{{
    "function_call": {{
        "name": "generate_dsl",
        "arguments": "{{\\"program\\": \\"add(10, 20), add(#0, 30), divide(#1, 3)\\"}}" 
    }}
}}
```
"""
    # --- END: New System Prompt ---

    messages = [{"role": "system", "content": system_prompt}]
    # The user role now just triggers the response based on the system prompt's context
    messages.append({"role": "user", "content": "Generate the DSL program based on the provided context."})

    try:
        model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        logger.info(f"Using OpenAI model: {model_name} for DSL generation")
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=300,
            functions=[DSL_GENERATION_FUNCTION_DEFINITION], # Use the schema defined outside
            function_call={"name": DSL_GENERATION_FUNCTION_NAME} # Force call
        )
        message = response.choices[0].message
        if message.get("function_call"):
            func_call = message["function_call"]
            if func_call.get("name") == DSL_GENERATION_FUNCTION_NAME:
                try:
                    args_str = func_call.get("arguments", "{}")
                    args = json.loads(args_str)
                    program = args.get("program")
                    if isinstance(program, str) and program.strip():
                        # Basic validation: Check for balanced parentheses as a sanity check
                        if program.count('(') == program.count(')'):
                            logger.info(f"Successfully generated DSL program: {program}")
                            return program.strip()
                        else:
                            logger.error(f"LLM generated DSL with unbalanced parentheses: {program}")
                            return ""
                    else:
                        logger.error(f"LLM generated invalid/empty program: {program}")
                        return ""
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode arguments JSON for DSL generation: {args_str}")
                    return ""
            else:
                logger.error(f"LLM called wrong function for DSL generation: {func_call.get('name')}")
                return ""
        else:
            logger.warning(f"LLM did not generate function call for DSL. Response: {message.get('content')}")
            return ""
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API call failed (DSL generation): {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error during LLM call (DSL generation): {e}")
        return ""

# --- END ADDITION ---

# --- START: New LLM-based Value Extraction Function --- 

# Define the function schema for the single value extraction LLM call
VALUE_EXTRACTION_FUNCTION_NAME = "report_extracted_value"
VALUE_EXTRACTION_FUNCTION_DEFINITION = {
    "name": VALUE_EXTRACTION_FUNCTION_NAME,
    "description": "Reports the single numerical value found in the evidence for the specified item.",
    "parameters": {
        "type": "object",
        "properties": {
            "value": {
                # Allow string initially to handle cases where LLM returns variations or indicates not found
                "type": ["string", "number", "null"],
                "description": "The extracted numerical value as a string (e.g., \"1,234.56\", \"(500)\", \"N/A\") or number. Use null or 'N/A' if the value cannot be found."
            }
        },
        "required": ["value"]
    }
}

def extract_single_value_with_llm(
    required_item: str, 
    evidence_chunks: List[Dict[str, str]]
) -> Optional[float | int]:
    """
    Uses an LLM function call to extract a single numerical value for the `required_item`
    from the `evidence_chunks`.

    Args:
        required_item: The specific item description (e.g., "Net Sales 2021").
        evidence_chunks: The list of evidence dictionaries.

    Returns:
        The extracted value as a float or int if found and valid, otherwise None.
    """
    logger.info(f"Attempting LLM extraction for: '{required_item}'")

    # --- Evidence pre-processing (same as generate_tool_call) ---
    processed_chunks_for_prompt = []
    # Limit evidence length to avoid excessive prompt size / cost
    # This could be tuned, maybe prioritize table chunks or chunks with the metric/period?
    MAX_EVIDENCE_TOKENS = 3000 # Rough estimate, depends on tokenizer
    current_token_count = 0
    
    for i, chunk in enumerate(evidence_chunks):
        chunk_id = chunk.get("chunk_id", f"unknown_{i}")
        text = chunk.get("text", "")
        prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\n{text}\n---"
        is_table_chunk = chunk_id.endswith("::table:full") and text.strip().startswith("Financial Table Data:")
        if is_table_chunk:
            # ... (Keep table parsing logic) ...
            md_content = text.split("\n\n", 1)[-1]
            parsed_table = parse_markdown_table(md_content)
            if parsed_table:
                try:
                    json_table_string = json.dumps(parsed_table, separators=(',', ':'))
                    prompt_text = f"Evidence {i+1} (ID: {chunk_id}):\n```json\n{json_table_string}\n```\n---"
                except Exception as json_err:
                    logger.error(f"Failed JSON dump for {chunk_id}: {json_err}")
        
        # Basic token estimation (split by space)
        chunk_tokens = len(prompt_text.split())
        if current_token_count + chunk_tokens < MAX_EVIDENCE_TOKENS:
            processed_chunks_for_prompt.append(prompt_text)
            current_token_count += chunk_tokens
        else:
            logger.warning(f"Skipping evidence chunk {chunk_id} for '{required_item}' extraction due to token limit.")
            # break # Option: Stop adding chunks once limit is reached
            
    if not processed_chunks_for_prompt:
        logger.error(f"No evidence chunks to process for '{required_item}' extraction after filtering.")
        return None
        
    evidence_text_for_prompt = "\n".join(processed_chunks_for_prompt)
    # --- End Evidence pre-processing ---

    # --- System Prompt for Value Extraction ---
    system_prompt = f"""\
You are a highly accurate data extraction bot. Your sole task is to find the specific numerical value corresponding to the requested financial item description, using the provided text and table evidence.

**Your Task:**

1.  Carefully read the `Item to Extract` description.
2.  Scan the `Relevant Evidence` provided.
3.  Locate the **exact numerical value** that corresponds to the description. The wording in the evidence might differ slightly from the description; focus on finding the correct number based on the metric, period, and context provided in the description.
4.  Handle different number formats (commas, parentheses for negatives, percentages).

**Output Requirements:**

*   **Format:** You MUST respond using the `report_extracted_value` function call.
*   **Value:**
    *   If you find the correct numerical value, provide it as a string or number in the `value` argument. **CRITICAL: The returned string must contain ONLY the numerical value and standard numeric symbols (commas ',', periods '.', hyphens/minuses '-', parentheses '()', percent signs '%'). It must NOT contain any letters, units (e.g., 'million'), currency names/symbols (e.g., 'USD', '$'), or descriptive words (e.g., 'approx'). Your output must be directly parsable by Python after basic cleaning.** Example of a GOOD value: "1,234.56".
    *   If the value is explicitly stated as not applicable or not available in the evidence, use null or "N/A" for the `value`.
    *   If you **cannot confidently locate** the specific value corresponding to the description in the evidence, use null or "N/A". Do NOT guess or calculate.
*   **Strictness:** Output ONLY the function call JSON. No explanations, conversation, or apologies.

**Example:**

*Item to Extract:*
```
Net Sales 2021
```
*Relevant Evidence:*
```
Evidence 1 (ID: doc1::table:full):
```json
[{{"Metric":"Net Sales","2022":"1,345.00","2021":"1,234.56","2020":"1,100.10"}}, ...]
```
---
Evidence 2 (ID: doc1::text:pre:5):
In 2021, net sales reached $1,234.56 million, an increase from the prior year.
---
```

*Correct Output:*
```json
{{
    "function_call": {{
        "name": "report_extracted_value",
        "arguments": "{{\\"value\\": \\"1,234.56\\"}}" # Note: String contains ONLY the number.
    }}
}}
```
"""

    # --- User Prompt --- 
    user_prompt = f"""Item to Extract:
```
{required_item}
```
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
        model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini") # Consider a potentially cheaper/faster model if latency is an issue
        logger.info(f"Using OpenAI model: {model_name} for value extraction of '{required_item}'")
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=150, # Should be small, only needs to return the value
            functions=[VALUE_EXTRACTION_FUNCTION_DEFINITION],
            function_call={"name": VALUE_EXTRACTION_FUNCTION_NAME}
        )

        message = response.choices[0].message
        if message.get("function_call"):
            func_call = message["function_call"]
            if func_call.get("name") == VALUE_EXTRACTION_FUNCTION_NAME:
                try:
                    args_str = func_call.get("arguments", "{}")
                    args = json.loads(args_str)
                    raw_value = args.get("value")

                    if raw_value is None or str(raw_value).strip().upper() == "N/A":
                        logger.warning(f"LLM indicated value for '{required_item}' was not found (N/A or null).")
                        return None

                    # Clean the extracted value
                    try:
                        cleaned_value_str = clean_numeric_string(str(raw_value))
                        if cleaned_value_str: # Ensure cleaning didn't result in empty string
                           numeric_value = float(cleaned_value_str) if '.' in cleaned_value_str else int(cleaned_value_str)
                           logger.info(f"Successfully extracted value for '{required_item}' via LLM: {numeric_value} (Raw: '{raw_value}')")
                           return numeric_value
                        else:
                           logger.error(f"Cleaning raw value '{raw_value}' for '{required_item}' resulted in empty string.")
                           return None
                    except ValueError as clean_err:
                        logger.error(f"Could not convert cleaned LLM value '{cleaned_value_str}' for '{required_item}' to number. Raw: '{raw_value}'. Error: {clean_err}")
                        return None

                except json.JSONDecodeError:
                    logger.error(f"Failed to decode arguments JSON for value extraction: {args_str}")
                    return None
            else:
                logger.error(f"LLM called wrong function for value extraction: {func_call.get('name')}")
                return None
        else:
            logger.warning(f"LLM did not return function call for value extraction of '{required_item}'. Response: {message.get('content')}")
            return None

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API call failed (value extraction for '{required_item}'): {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during LLM call (value extraction for '{required_item}'): {e}")
        return None

# --- END: New LLM-based Value Extraction Function ---
