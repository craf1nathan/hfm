"""
Fonctions utilitaires partagées.
"""

import numpy as np
from datetime import datetime

def safe_divide(a, b, default=0.0):
    """Division sécurisée."""
    return a / b if abs(b) > 1e-12 else default

def log_section(title: str):
    """Print une section de log."""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")

def log_result(label: str, value, format_spec=None):
    """Log un résultat formaté."""
    if format_spec:
        print(f"  {label:<30s}: {value:{format_spec}}")
    else:
        print(f"  {label:<30s}: {value}")

class ProgressBar:
    """Simple progress bar."""
    def __init__(self, total: int, label: str = "Progress"):
        self.total = total
        self.label = label
        self.current = 0
    
    def update(self, step=1):
        self.current += step
        pct = self.current / self.total
        bar_length = 40
        filled = int(bar_length * pct)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r{self.label}: [{bar}] {pct:.1%}", end='', flush=True)
        if pct >= 1.0:
            print()
