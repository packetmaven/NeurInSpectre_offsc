#!/usr/bin/env python3
"""
NeurInSpectre Package Main Entry Point
Enables execution via: python -m neurinspectre <command>
"""

import sys

def main():
    """Main entry point for neurinspectre package"""
    # Delegate to Click-based CLI (with legacy fallback)
    from neurinspectre.cli.main import main as cli_main
    return cli_main()

if __name__ == '__main__':
    sys.exit(main()) 