#!/usr/bin/env python3
"""
Protein Stability DoE Suite
Entry point for the merged application

Author: Milton F. Villegas
Email: miltonfvillegas@gmail.com
Version: 1.0.0
"""

import sys
from gui.main_window import ProteinDoESuite


def main():
    """Launch the application"""
    app = ProteinDoESuite()
    app.mainloop()


if __name__ == '__main__':
    main()
