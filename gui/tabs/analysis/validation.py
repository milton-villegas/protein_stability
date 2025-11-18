#!/usr/bin/env python3
"""
Constraint validation functions for DoE Analysis.
Based on best practices from DOE software (Minitab, JMP, Design-Expert).
"""


def validate_constraint(response_name, direction, constraint, data_min, data_max, total_experiments):
    """
    Comprehensive constraint validation with severity levels: error, warning, info

    Returns list of dicts with: severity, category, message, should_stop, experiments_meeting
    """
    results = []

    # Check MIN constraint
    if 'min' in constraint:
        min_val = constraint['min']
        result = _validate_min_constraint(response_name, direction, min_val, data_min, data_max, total_experiments)
        if result:
            results.append(result)

    # Check MAX constraint
    if 'max' in constraint:
        max_val = constraint['max']
        result = _validate_max_constraint(response_name, direction, max_val, data_min, data_max, total_experiments)
        if result:
            results.append(result)

    # Check for invalid range (Min > Max)
    if 'min' in constraint and 'max' in constraint:
        if constraint['min'] > constraint['max']:
            results.append({
                'severity': 'error',
                'category': 'invalid_range',
                'message': f"{response_name}: Min ({constraint['min']}) > Max ({constraint['max']}) - Invalid constraint range",
                'should_stop': True,
                'experiments_meeting': 0
            })

    return results


def _validate_min_constraint(response_name, direction, min_val, data_min, data_max, total_experiments):
    """Validate MIN constraint"""
    # Case 1: Min below all data (useless)
    if min_val < data_min:
        return {
            'severity': 'warning',
            'category': 'useless_min',
            'message': f"{response_name} Min={min_val:.2f} has no effect (all data is already >= {data_min:.2f})",
            'detail': f"Your data range is {data_min:.2f} to {data_max:.2f}. All {total_experiments} experiments already meet this Min constraint.",
            'should_stop': False,
            'experiments_meeting': total_experiments
        }

    # Case 2: Min within data range
    elif data_min <= min_val <= data_max:
        range_covered = (data_max - min_val) / (data_max - data_min)
        experiments_meeting = max(1, int(total_experiments * range_covered))

        if experiments_meeting < total_experiments * 0.2:
            return {
                'severity': 'info',
                'category': 'restrictive_min',
                'message': f"{response_name} Min={min_val:.2f} is restrictive (~{experiments_meeting}/{total_experiments} experiments meet it)",
                'should_stop': False,
                'experiments_meeting': experiments_meeting
            }
        else:
            return {
                'severity': 'info',
                'category': 'valid_min',
                'message': f"{response_name} Min={min_val:.2f} is valid (~{experiments_meeting}/{total_experiments} experiments meet it)",
                'should_stop': False,
                'experiments_meeting': experiments_meeting
            }

    # Case 3: Min above all data
    else:
        if direction == 'minimize':
            return {
                'severity': 'error',
                'category': 'contradiction',
                'message': f"{response_name}: CONTRADICTION - MINIMIZE with Min={min_val:.2f} above all data ({data_max:.2f})",
                'detail': f"You want to MINIMIZE (get lowest value) but require Min >= {min_val:.2f}, which is higher than your best data ({data_max:.2f}). This is impossible!",
                'should_stop': True,
                'experiments_meeting': 0
            }
        else:
            return {
                'severity': 'warning',
                'category': 'no_data_min',
                'message': f"{response_name} Min={min_val:.2f} is above all data ({data_max:.2f}) - will be ignored",
                'detail': f"No experiments meet this constraint. Analysis will proceed without it.",
                'should_stop': False,
                'experiments_meeting': 0
            }


def _validate_max_constraint(response_name, direction, max_val, data_min, data_max, total_experiments):
    """Validate MAX constraint"""
    # Case 1: Max below all data
    if max_val < data_min:
        if direction == 'maximize':
            return {
                'severity': 'error',
                'category': 'contradiction',
                'message': f"{response_name}: CONTRADICTION - MAXIMIZE with Max={max_val:.2f} below all data ({data_min:.2f})",
                'detail': f"You want to MAXIMIZE (get highest value) but require Max <= {max_val:.2f}, which is lower than your worst data ({data_min:.2f}). This is impossible!",
                'should_stop': True,
                'experiments_meeting': 0
            }
        else:
            return {
                'severity': 'warning',
                'category': 'no_data_max',
                'message': f"{response_name} Max={max_val:.2f} is below all data ({data_min:.2f}) - will be ignored",
                'detail': f"No experiments meet this constraint. Analysis will proceed without it.",
                'should_stop': False,
                'experiments_meeting': 0
            }

    # Case 2: Max within data range
    elif data_min <= max_val <= data_max:
        range_covered = (max_val - data_min) / (data_max - data_min)
        experiments_meeting = max(1, int(total_experiments * range_covered))

        if experiments_meeting < total_experiments * 0.2:
            return {
                'severity': 'info',
                'category': 'restrictive_max',
                'message': f"{response_name} Max={max_val:.2f} is restrictive (~{experiments_meeting}/{total_experiments} experiments meet it)",
                'should_stop': False,
                'experiments_meeting': experiments_meeting
            }
        else:
            return {
                'severity': 'info',
                'category': 'valid_max',
                'message': f"{response_name} Max={max_val:.2f} is valid (~{experiments_meeting}/{total_experiments} experiments meet it)",
                'should_stop': False,
                'experiments_meeting': experiments_meeting
            }

    # Case 3: Max above all data (useless)
    else:
        return {
            'severity': 'warning',
            'category': 'useless_max',
            'message': f"{response_name} Max={max_val:.2f} has no effect (all data is already <= {data_max:.2f})",
            'detail': f"Your data range is {data_min:.2f} to {data_max:.2f}. All {total_experiments} experiments already meet this Max constraint.",
            'should_stop': False,
            'experiments_meeting': total_experiments
        }
