"""
Models and validation functions for the Designer tab.

This module contains the FactorModel class for managing experimental factors
and their stock concentrations, along with validation functions for user input.

The FactorModel supports full factorial experimental design by tracking:
- Factor names and their discrete levels
- Optional stock concentrations for volume calculations
- Total combination counts for experimental planning

Validation functions are designed for use with Tkinter Entry widget validation,
following the validatecommand protocol.
"""

from typing import Dict, List, Optional


def validate_numeric_input(action: str, char: str, entry_value: str) -> bool:
    """
    Validate numeric input for Tkinter Entry widgets.

    Allows digits, decimal points, minus signs, and commas for entering
    multiple numeric values in a single field.

    Args:
        action: Tkinter validation action ('0' for delete, '1' for insert)
        char: Character being inserted
        entry_value: Current value in the entry widget

    Returns:
        True if the input is valid, False otherwise

    Examples:
        >>> validate_numeric_input('1', '5', '')
        True
        >>> validate_numeric_input('1', '.', '3')
        True
        >>> validate_numeric_input('1', ',', '3.5')
        True
        >>> validate_numeric_input('1', 'a', '3.5')
        False
    """
    # Allow all deletions
    if action == '0':
        return True
    # Allow empty
    if char == '':
        return True
    # Allow digits, decimal point, negative sign, and comma for multiple entries
    return char.isdigit() or char in '.-,'


def validate_single_numeric_input(action: str, char: str, entry_value: str) -> bool:
    """
    Validate single numeric input for Tkinter Entry widgets.

    Similar to validate_numeric_input but does not allow commas,
    restricting input to a single numeric value.

    Args:
        action: Tkinter validation action ('0' for delete, '1' for insert)
        char: Character being inserted
        entry_value: Current value in the entry widget

    Returns:
        True if the input is valid, False otherwise

    Examples:
        >>> validate_single_numeric_input('1', '5', '')
        True
        >>> validate_single_numeric_input('1', '-', '')
        True
        >>> validate_single_numeric_input('1', ',', '3.5')
        False
    """
    if action == '0':
        return True

    if char == '':
        return True

    return char.isdigit() or char in '.-'


def validate_alphanumeric_input(action: str, char: str, entry_value: str) -> bool:
    """
    Validate alphanumeric input for Tkinter Entry widgets.

    Allows letters, digits, spaces, and common punctuation characters
    suitable for factor names and descriptions.

    Args:
        action: Tkinter validation action ('0' for delete, '1' for insert)
        char: Character being inserted
        entry_value: Current value in the entry widget

    Returns:
        True if the input is valid, False otherwise

    Examples:
        >>> validate_alphanumeric_input('1', 'N', '')
        True
        >>> validate_alphanumeric_input('1', ' ', 'NaCl')
        True
        >>> validate_alphanumeric_input('1', '(', 'pH')
        True
        >>> validate_alphanumeric_input('1', '@', 'test')
        False
    """
    # Allow all deletions
    if action == '0':
        return True
    # Allow empty
    if char == '':
        return True
    # Allow letters, digits, spaces, hyphens, parentheses, and commas
    return char.isalnum() or char in ' -(),.'


class FactorModel:
    """
    Model for managing experimental factors and their stock concentrations.

    This class provides a data model for storing and manipulating factors
    in a full factorial experimental design. Each factor has a name,
    a list of discrete levels, and an optional stock concentration.

    The model supports:
    - Adding, updating, and removing factors
    - Tracking stock concentrations for volume calculations
    - Computing total factorial combinations

    Attributes:
        _factors: Internal dictionary mapping factor names to level lists
        _stock_concs: Internal dictionary mapping factor names to stock concentrations

    Example:
        >>> model = FactorModel()
        >>> model.add_factor('NaCl', ['0', '50', '100', '150'], stock_conc=1000.0)
        >>> model.add_factor('pH', ['6.0', '7.0', '8.0'])
        >>> model.total_combinations()
        12
        >>> model.get_stock_conc('NaCl')
        1000.0
    """

    def __init__(self):
        """Initialize empty FactorModel with no factors or stock concentrations."""
        self._factors: Dict[str, List[str]] = {}
        self._stock_concs: Dict[str, float] = {}

    def add_factor(self, name: str, levels: List[str], stock_conc: Optional[float] = None) -> None:
        """
        Add a new factor to the model.

        Creates a new factor with the specified name and levels. If a factor
        with the same name already exists, it will be overwritten.

        Args:
            name: Factor name (e.g., 'NaCl', 'buffer pH'). Will be stripped
                  of leading/trailing whitespace.
            levels: List of factor levels as strings (e.g., ['0', '50', '100'])
            stock_conc: Optional stock concentration for volume calculations.
                        Units should be consistent across all factors.

        Raises:
            ValueError: If name is empty after stripping whitespace
            ValueError: If levels list is empty

        Example:
            >>> model = FactorModel()
            >>> model.add_factor('NaCl', ['0', '100', '200'], stock_conc=1000.0)
        """
        name = name.strip()
        if not name:
            raise ValueError("Factor name cannot be empty.")
        if not levels:
            raise ValueError("At least one level is required.")
        self._factors[name] = list(levels)
        if stock_conc is not None:
            self._stock_concs[name] = stock_conc

    def update_factor(self, name: str, levels: List[str], stock_conc: Optional[float] = None) -> None:
        """
        Update an existing factor's levels and/or stock concentration.

        Replaces the levels for an existing factor. The stock concentration
        is only updated if a new value is provided.

        Args:
            name: Factor name to update (must already exist)
            levels: New list of factor levels
            stock_conc: Optional new stock concentration. If None, existing
                        stock concentration is preserved.

        Raises:
            ValueError: If factor with given name doesn't exist
            ValueError: If levels list is empty

        Example:
            >>> model = FactorModel()
            >>> model.add_factor('NaCl', ['0', '100'])
            >>> model.update_factor('NaCl', ['0', '50', '100', '150'])
        """
        if name not in self._factors:
            raise ValueError(f"Factor '{name}' does not exist.")
        if not levels:
            raise ValueError("At least one level is required.")
        self._factors[name] = list(levels)
        if stock_conc is not None:
            self._stock_concs[name] = stock_conc

    def remove_factor(self, name: str) -> None:
        """
        Remove a factor and its stock concentration from the model.

        Silently ignores if the factor doesn't exist.

        Args:
            name: Factor name to remove

        Example:
            >>> model = FactorModel()
            >>> model.add_factor('NaCl', ['0', '100'])
            >>> model.remove_factor('NaCl')
            >>> model.get_factors()
            {}
        """
        if name in self._factors:
            del self._factors[name]
        if name in self._stock_concs:
            del self._stock_concs[name]

    def get_factors(self) -> Dict[str, List[str]]:
        """
        Get all factors and their levels.

        Returns a deep copy to prevent external modification of internal state.

        Returns:
            Dictionary mapping factor names to lists of level strings.
            Modifying the returned dictionary will not affect the model.

        Example:
            >>> model = FactorModel()
            >>> model.add_factor('NaCl', ['0', '100'])
            >>> model.get_factors()
            {'NaCl': ['0', '100']}
        """
        return {k: list(v) for k, v in self._factors.items()}

    def get_stock_conc(self, name: str) -> Optional[float]:
        """
        Get stock concentration for a specific factor.

        Args:
            name: Factor name

        Returns:
            Stock concentration as float, or None if not set for this factor

        Example:
            >>> model = FactorModel()
            >>> model.add_factor('NaCl', ['0', '100'], stock_conc=1000.0)
            >>> model.get_stock_conc('NaCl')
            1000.0
            >>> model.get_stock_conc('pH') is None
            True
        """
        return self._stock_concs.get(name)

    def get_all_stock_concs(self) -> Dict[str, float]:
        """
        Get all stock concentrations.

        Returns a copy to prevent external modification of internal state.

        Returns:
            Dictionary mapping factor names to their stock concentrations.
            Only factors with stock concentrations set are included.

        Example:
            >>> model = FactorModel()
            >>> model.add_factor('NaCl', ['0', '100'], stock_conc=1000.0)
            >>> model.add_factor('pH', ['6.0', '7.0'])  # No stock conc
            >>> model.get_all_stock_concs()
            {'NaCl': 1000.0}
        """
        return dict(self._stock_concs)

    def clear(self) -> None:
        """
        Remove all factors and stock concentrations from the model.

        Resets the model to its initial empty state.

        Example:
            >>> model = FactorModel()
            >>> model.add_factor('NaCl', ['0', '100'])
            >>> model.clear()
            >>> model.total_combinations()
            0
        """
        self._factors.clear()
        self._stock_concs.clear()

    def total_combinations(self) -> int:
        """
        Calculate total number of full factorial combinations.

        Computes the product of all factor level counts, representing
        the total number of experimental conditions in a full factorial design.

        Returns:
            Product of all factor level counts. Returns 0 if no factors
            have been added to the model.

        Example:
            >>> model = FactorModel()
            >>> model.add_factor('NaCl', ['0', '50', '100'])  # 3 levels
            >>> model.add_factor('pH', ['6.0', '7.0'])        # 2 levels
            >>> model.total_combinations()
            6
        """
        if not self._factors:
            return 0
        # Multiply all factor level counts (full factorial)
        result = 1
        for levels in self._factors.values():
            result *= len(levels)
        return result
