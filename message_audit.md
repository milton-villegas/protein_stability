# Message Standardization Audit

## Current State Analysis

### Designer Tab Messages

#### ✅ **Well Formatted (Good Examples)**
```python
# Good: Clear, specific, with solution
messagebox.showerror("Missing Stock Concentration",
    "Stock concentration is required for this factor.")

# Good: Validation with explanation
messagebox.showerror("Invalid Value",
    f"Invalid pH value: {value}\n\n"
    f"pH must be between 1.0 and 14.0\n"
    f"(Typical range: 2-12 for most buffers)")

# Good: Error with context and solution
messagebox.showerror("Too Many Combinations",
    f"Design too large: {estimated_size} combinations exceeds "
    f"{MAX_TOTAL_WELLS} wells (4 plates of 96 wells each).\n\n"
    f"Solutions:\n"
    f"  1. Reduce the number of factor levels\n"
    f"  2. Use fewer factors\n"
    f"  3. Use Latin Hypercube Sampling (LHS) design instead")
```

#### ❌ **Inconsistent/Poor Formatting**
```python
# Bad: Too generic, no context
messagebox.showerror("Error", str(e))

# Bad: Inconsistent title case
messagebox.showinfo("Already Added", ...)  # Title case
messagebox.showinfo("No Selection", ...)   # Title case
messagebox.showerror("Invalid Name", ...)  # Title case
messagebox.showerror("Error", ...)          # Generic

# Bad: No newline separation
messagebox.showerror("Export Failed", f"Error during export:\n\n{str(e)}")
# Should be: "Export Failed", f"Error during export:\n\n{str(e)}")

# Bad: Inconsistent success messages
messagebox.showinfo("Export Complete", f"Files exported to:\n{directory}")
# vs
messagebox.showinfo("Success", ...)
```

### Analyzer Tab Messages

#### ✅ **Good Examples**
```python
# Good: Clear error with context
messagebox.showerror("Error",
    "Excel file must have a column named 'Response'\n\n"
    "Please rename your response column to 'Response' and try again.")
```

#### ❌ **Poor Examples**
```python
# Bad: Generic title
messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
messagebox.showerror("Error", f"Export failed:\n{str(e)}")

# Bad: Inconsistent phrasing
messagebox.showinfo("Success", ...)
messagebox.showinfo("Export Complete", ...)
```

---

## Issues Found

### 1. **Inconsistent Titles**
- Generic "Error" used 15+ times
- Mix of Title Case and sentence case
- No clear naming convention

### 2. **Inconsistent Message Structure**
- Some have solutions, some don't
- Varying levels of detail
- Inconsistent use of \n\n for spacing

### 3. **Poor Error Context**
- Many just show `str(e)` without explanation
- No indication of what the user should do
- Missing "why this happened" context

### 4. **Mixed Message Types**
- showerror vs showwarning vs showinfo used inconsistently
- No clear guideline for which to use when

---

## Proposed Standards

### **Message Type Guidelines**

| Type | When to Use | Example |
|------|-------------|---------|
| **showerror** | User action failed, cannot proceed | Missing required field, invalid input |
| **showwarning** | Action succeeded but with issues | File loaded but missing columns |
| **showinfo** | Success confirmation or guidance | Export complete, help text |

### **Title Standards**

| Category | Title Format | Examples |
|----------|-------------|----------|
| **Validation Errors** | "{Field} Invalid" or "Invalid {Field}" | "Invalid pH Value", "Stock Concentration Invalid" |
| **Missing Data** | "Missing {Item}" | "Missing Stock Concentration", "Missing Response Column" |
| **Import/Export** | "{Action} Failed" or "{Action} Complete" | "Export Failed", "Import Complete" |
| **User Guidance** | Descriptive phrase | "No Selection", "Already Exists" |

### **Message Structure Template**

```python
# Template for error messages
messagebox.showerror(
    "Title (Specific, Not Generic 'Error')",
    f"What happened: {specific_issue}\n\n"
    f"Why it happened: {explanation}\n\n"
    f"What to do:\n"
    f"  1. {solution_step_1}\n"
    f"  2. {solution_step_2}"
)

# Template for success messages
messagebox.showinfo(
    "{Action} Complete",
    f"Success! {what_was_done}\n\n"
    f"Location: {where_output_is}\n"
    f"Next steps: {optional_guidance}"
)

# Template for warnings
messagebox.showwarning(
    "{Action} Warning",
    f"Action completed with warnings:\n\n"
    f"{warning_description}\n\n"
    f"Impact: {what_this_means}\n"
    f"Recommendation: {what_to_do}"
)
```

---

## Standardization Examples

### **Before vs After**

#### Example 1: Generic Error
```python
# ❌ BEFORE
messagebox.showerror("Error", str(e))

# ✅ AFTER
messagebox.showerror(
    "Operation Failed",
    f"The operation could not be completed:\n\n"
    f"{str(e)}\n\n"
    f"Please check your inputs and try again."
)
```

#### Example 2: Export Success
```python
# ❌ BEFORE (Inconsistent)
messagebox.showinfo("Export Complete", f"Files exported to:\n{directory}")
messagebox.showinfo("Success", "Files exported successfully")

# ✅ AFTER (Consistent)
messagebox.showinfo(
    "Export Complete",
    f"Design files successfully exported!\n\n"
    f"Location: {directory}\n\n"
    f"Files created:\n"
    f"  • {excel_file}\n"
    f"  • {csv_file}"
)
```

#### Example 3: Validation Error
```python
# ❌ BEFORE
messagebox.showerror("Invalid Value", error_msg)

# ✅ AFTER
messagebox.showerror(
    "Invalid pH Value",
    f"pH value {value} is out of range.\n\n"
    f"Valid range: 1.0 to 14.0\n"
    f"Typical range: 2.0 to 12.0 for most buffers\n\n"
    f"Please enter a valid pH value."
)
```

#### Example 4: Missing Dependency
```python
# ❌ BEFORE
raise ImportError("pyDOE3 is required for Latin Hypercube Sampling. "
                  "Install with: pip install pyDOE3")

# ✅ AFTER
messagebox.showerror(
    "Missing Dependency",
    "pyDOE3 library is required for Latin Hypercube Sampling.\n\n"
    "Installation:\n"
    "  pip install pyDOE3\n\n"
    "Or install all optional dependencies:\n"
    "  pip install -r requirements-designer.txt"
)
```

---

## Implementation Plan

### Phase 1: Core Messages (High Priority)
1. Export/Import success/failure messages
2. Validation errors (pH, concentrations, etc.)
3. Missing data errors
4. Design generation errors

### Phase 2: User Guidance (Medium Priority)
5. Help messages (tooltips, guides)
6. Confirmation dialogs
7. Warning messages

### Phase 3: Developer Messages (Low Priority)
8. Exception messages (ValueError, ImportError)
9. Debug/logging messages

---

## Proposed Message Categories

### 1. **Validation Errors**
```python
"Invalid {Field}"           # Invalid pH Value, Invalid Stock Concentration
"Invalid {Action}"          # Invalid Input, Invalid Range
"{Field} Out of Range"      # pH Out of Range
```

### 2. **Missing Data**
```python
"Missing {Item}"            # Missing Stock Concentration, Missing Response Column
"No {Item} Found"           # No Factors Found, No Data Loaded
"{Item} Required"           # Stock Concentration Required
```

### 3. **Import/Export**
```python
"{Action} Complete"         # Export Complete, Import Complete
"{Action} Failed"           # Export Failed, Import Failed
"Loading {Item}"            # Loading Data
```

### 4. **Dependencies**
```python
"Missing Dependency"        # For all import errors
"Library Required"          # Alternative
```

### 5. **User Actions**
```python
"Confirm {Action}"          # Confirm Delete, Confirm Clear All
"No Selection"              # When nothing selected
"Already Exists"            # Duplicate items
```

### 6. **Design Errors**
```python
"Design Too Large"
"Design Invalid"
"Insufficient Data"
```

---

## Formatting Standards

### **Spacing**
- Use `\n\n` (double newline) between sections
- Use `\n` (single newline) within lists
- Always end with a newline before closing quote

### **Structure**
1. **Line 1**: What happened (1 sentence)
2. **Line 2** (blank line)
3. **Line 3**: Why it happened or more details
4. **Line 4** (blank line)
5. **Line 5+**: What to do (numbered or bulleted list)

### **Tone**
- Professional but friendly
- Action-oriented (tell user what to do)
- Specific, not vague
- Empathetic (acknowledge frustration)

### **Length**
- Keep total message under 200 characters if possible
- Max 5 lines of text
- Use bullet points for multiple solutions

---

## Code Standards

### **Good Practices**
```python
# ✅ Use specific titles
messagebox.showerror("Invalid pH Value", ...)

# ✅ Provide context and solution
messagebox.showerror(
    "Export Failed",
    f"Could not save files to {path}\n\n"
    f"Reason: {error}\n\n"
    f"Solutions:\n"
    f"  1. Check file permissions\n"
    f"  2. Ensure directory exists\n"
    f"  3. Close any open Excel files"
)

# ✅ Use appropriate message type
messagebox.showwarning("Design Large", ...)  # Warning, not error
messagebox.showinfo("Export Complete", ...)   # Success
messagebox.showerror("Invalid Input", ...)    # Blocking error
```

### **Bad Practices to Avoid**
```python
# ❌ Generic title
messagebox.showerror("Error", ...)

# ❌ Just exception text
messagebox.showerror("Error", str(e))

# ❌ No spacing
messagebox.showerror("Title", f"Line1\nLine2\nLine3")

# ❌ No solution
messagebox.showerror("Failed", "Operation failed")

# ❌ Wrong message type
messagebox.showerror("Success", ...)  # Use showinfo for success
```

---

## Next Steps

1. Create message utility module (`gui/utils/messages.py`)
2. Define standard message templates
3. Replace all messagebox calls with standardized versions
4. Add unit tests for message formatting
5. Document in developer guide
