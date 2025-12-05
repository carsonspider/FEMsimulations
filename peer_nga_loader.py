#!/usr/bin/env python3
"""
PEER NGA Database Loader

Loads earthquake ground motion records from PEER NGA-West2 and NGA-East databases.

PEER NGA File Format:
====================
PEER NGA files are ASCII text files with the following structure:

Line 1: Record name/identifier
Line 2: Number of data points, time step (dt), units
Line 3+: Acceleration values (one per line)

Example:
--------
NORTHRIDGE/SYL090
4000, 0.005000, ACCEL G
-0.001234
0.002345
...

References:
-----------
1. Ancheta, T. D., et al. (2014). "NGA-West2 Database." 
   Earthquake Spectra, 30(3), 989-1005.
   
2. PEER Ground Motion Database: https://peer.berkeley.edu/peer-strong-ground-motion-databases
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import re


class PEERNGAReader:
    """Reader for PEER NGA earthquake record files"""
    
    @staticmethod
    def read_peer_nga_file(filepath: Path) -> Dict:
        """
        Read PEER NGA format earthquake record
        
        Parameters:
        -----------
        filepath : Path
            Path to PEER NGA ASCII file
            
        Returns:
        --------
        Dict with keys:
            - 'name': Record identifier
            - 'time': Time array (seconds)
            - 'acceleration': Acceleration array (g or m/s²)
            - 'dt': Time step (seconds)
            - 'npts': Number of data points
            - 'units': Units ('g' or 'm/s²')
            - 'duration': Total duration (seconds)
            - 'pga': Peak ground acceleration
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"PEER NGA file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Line 1: Record name
        name = lines[0].strip()
        
        # Line 2: npts, dt, units
        header_line = lines[1].strip()
        # Parse: "4000, 0.005000, ACCEL G" or "4000, 0.005000, ACCEL M/S2"
        match = re.match(r'(\d+)\s*,\s*([\d.]+)\s*,\s*ACCEL\s+([GM])', header_line, re.IGNORECASE)
        if not match:
            # Try alternative format
            match = re.match(r'(\d+)\s*,\s*([\d.]+)\s*,\s*([GM])', header_line, re.IGNORECASE)
        
        if not match:
            raise ValueError(f"Could not parse PEER NGA header: {header_line}")
        
        npts = int(match.group(1))
        dt = float(match.group(2))
        unit_char = match.group(3).upper()
        
        # Determine units
        if unit_char == 'G':
            units = 'g'
            conversion_factor = 9.81  # Convert g to m/s²
        elif unit_char == 'M':
            units = 'm/s²'
            conversion_factor = 1.0
        else:
            units = 'g'  # Default assumption
            conversion_factor = 9.81
        
        # Read acceleration data (lines 3 onwards)
        acceleration = []
        for line in lines[2:]:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    accel_val = float(line)
                    acceleration.append(accel_val)
                except ValueError:
                    continue  # Skip non-numeric lines
        
        acceleration = np.array(acceleration)
        
        # Convert to m/s² if needed
        if units == 'g':
            acceleration = acceleration * conversion_factor
            units = 'm/s²'
        
        # Truncate or pad to match npts
        if len(acceleration) > npts:
            acceleration = acceleration[:npts]
        elif len(acceleration) < npts:
            # Pad with zeros
            acceleration = np.pad(acceleration, (0, npts - len(acceleration)), 'constant')
        
        # Create time array
        time = np.arange(0, npts * dt, dt)
        if len(time) > len(acceleration):
            time = time[:len(acceleration)]
        
        # Compute metrics
        duration = time[-1] - time[0] if len(time) > 1 else 0.0
        pga = np.max(np.abs(acceleration))
        
        return {
            'name': name,
            'time': time,
            'acceleration': acceleration,
            'dt': dt,
            'npts': npts,
            'units': units,
            'duration': duration,
            'pga': pga,
            'pga_g': pga / 9.81  # Also in g units
        }
    
    @staticmethod
    def scale_to_pga(record: Dict, target_pga: float) -> Dict:
        """
        Scale earthquake record to target PGA
        
        Parameters:
        -----------
        record : Dict
            PEER NGA record dictionary
        target_pga : float
            Target PGA in m/s²
            
        Returns:
        --------
        Scaled record dictionary
        """
        scale_factor = target_pga / record['pga']
        
        scaled_record = record.copy()
        scaled_record['acceleration'] = record['acceleration'] * scale_factor
        scaled_record['pga'] = target_pga
        scaled_record['pga_g'] = target_pga / 9.81
        scaled_record['name'] = f"{record['name']}_scaled_{target_pga/9.81:.2f}g"
        
        return scaled_record


def load_peer_nga_record(filepath: str, scale_pga: Optional[float] = None) -> Dict:
    """
    Convenience function to load PEER NGA record
    
    Parameters:
    -----------
    filepath : str
        Path to PEER NGA file
    scale_pga : float, optional
        Target PGA in m/s² (if None, uses original PGA)
        
    Returns:
    --------
    Dict with earthquake record data
    """
    filepath = Path(filepath)
    record = PEERNGAReader.read_peer_nga_file(filepath)
    
    if scale_pga is not None:
        record = PEERNGAReader.scale_to_pga(record, scale_pga)
    
    return record


# Example: Download and use PEER NGA records
# PEER NGA records can be downloaded from:
# https://peer.berkeley.edu/peer-strong-ground-motion-databases
#
# Example usage:
# record = load_peer_nga_record("path/to/peer_record.txt", scale_pga=0.5*9.81)  # Scale to 0.5g

