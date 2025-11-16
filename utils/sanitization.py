"""
Factor name sanitization and matching utilities
Handles conversion between display names and internal names
"""
import re
from typing import Optional

def sanitize_name(name: str) -> str:
    """
    Convert display name to Python-safe identifier

    Examples:
        "Buffer pH" → "buffer_ph"
        "NaCl (mM)" → "nacl_mm"
        "Glycerol (%)" → "glycerol"

    Args:
        name: Original factor name

    Returns:
        Sanitized name safe for Python identifiers
    """
    # Remove special characters, replace spaces with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_').lower()
    # Ensure doesn't start with number
    if sanitized and sanitized[0].isdigit():
        sanitized = 'x' + sanitized
    return sanitized

def smart_factor_match(display_name: str) -> str:
    """
    Intelligently match display name to internal factor name
    Handles any factor format by stripping units and normalizing

    Examples:
        "NaCl (mM)" → "nacl"
        "Detergent (%)" → "detergent_concentration"
        "Buffer Conc (mM)" → "buffer_concentration"
        "Buffer pH" → "buffer pH"
        "Reducing Agent (mM)" → "reducing_agent_concentration"

    Args:
        display_name: Factor display name from Excel/GUI

    Returns:
        Internal factor name
    """
    # Remove leading/trailing whitespace
    name = display_name.strip()

    # Special case: Buffer pH (keep as-is)
    if "Buffer pH" in name or "buffer pH" in name:
        return "buffer pH"

    # Extract base name by removing units in parentheses
    # "NaCl (mM)" → "NaCl"
    if '(' in name:
        base_name = name.split('(')[0].strip()
    else:
        base_name = name

    # Normalize: lowercase, replace spaces/hyphens with underscores
    normalized = base_name.lower().replace(' ', '_').replace('-', '_')

    # Handle special cases for concentration factors
    if "buffer_conc" in normalized or "buffer conc" in base_name.lower():
        return "buffer_concentration"
    elif "detergent" in base_name.lower() and ("%" in name or "conc" in base_name.lower()):
        return "detergent_concentration"
    elif "reducing_agent" in normalized or "reducing agent" in base_name.lower():
        if "mM" in name or "conc" in base_name.lower():
            return "reducing_agent_concentration"
        else:
            return "reducing_agent"

    # For everything else, return the normalized name
    # Examples: "NaCl" → "nacl", "Zinc" → "zinc", "Glycerol" → "glycerol"
    return normalized

def smart_column_match(column_name: str) -> Optional[str]:
    """
    Intelligently match Excel column name to internal factor name
    Similar to smart_factor_match but handles edge cases for BO export

    Examples:
        "NaCl (mM)" → "nacl"
        "Detergent" → "detergent" (categorical)
        "Detergent (%)" → "detergent_concentration"
        "Buffer pH" → "buffer pH"
        "Reducing Agent" → "reducing_agent" (categorical)
        "Reducing Agent (mM)" → "reducing_agent_concentration"

    Args:
        column_name: Column header from Excel file

    Returns:
        Internal factor name or None if invalid
    """
    # Handle None or empty column names
    if column_name is None or (isinstance(column_name, str) and not column_name.strip()):
        return None

    name = column_name.strip()

    # Special case: Buffer pH (keep as-is)
    if "Buffer pH" in name or "buffer pH" in name:
        return "buffer pH"

    # Extract base name by removing units
    if '(' in name:
        base_name = name.split('(')[0].strip()
    else:
        base_name = name

    # Normalize
    normalized = base_name.lower().replace(' ', '_').replace('-', '_')

    # Handle concentration suffixes
    if "buffer_conc" in normalized or "buffer conc" in base_name.lower():
        return "buffer_concentration"
    elif "detergent" in base_name.lower():
        # Check if it has units (concentration) or is categorical (name)
        if "%" in column_name or "conc" in base_name.lower():
            return "detergent_concentration"
        else:
            return "detergent"
    elif "reducing_agent" in normalized or "reducing agent" in base_name.lower():
        # Check if it has units (concentration) or is categorical (name)
        if "mM" in column_name or "conc" in base_name.lower():
            return "reducing_agent_concentration"
        else:
            return "reducing_agent"

    # Default: return normalized name
    return normalized

def get_display_name(internal_name: str, available_factors: dict) -> str:
    """
    Get display name from internal name using lookup dict

    Args:
        internal_name: Internal factor name (e.g., "buffer_concentration")
        available_factors: Dict mapping internal → display names

    Returns:
        Display name or internal name if not found
    """
    return available_factors.get(internal_name, internal_name)
