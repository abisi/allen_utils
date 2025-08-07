#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_analysis
@file: allen_utils.py
@time: 1/21/2025 3:36 PM
"""

# Imports
import os
import json
import pandas as pd
import numpy as np
import re


def get_cortical_areas():
    """
    Retrieve a list of cortical(++) area acronyms.
    :return: List of cortical area acronyms
    """
    return [
        'FRP', 'MOp', 'MOs', 'SSp-bfd', 'SSp-m', 'SSp-ul', 'SSp-ll', 'SSp-un', 'SSp-n', 'SSp-tr',
        'SSs', 'AUDp', 'AUDd', 'AUDv', 'ACA', 'ACAv', 'ACAd', 'VISa', 'VISp', 'VISam', 'VISl',
        'VISpm', 'VISrl', 'VISal', 'PL', 'ILA', 'ORB', 'RSP', 'RSPv', 'RSPd','RSPagl', 'TT', 'SCm',
        'SCsg', 'SCzo', 'SCiw', 'SCop', 'SCs', 'ORBm', 'ORBl', 'ORBvl', 'AId',
        'AIv', 'AIp', 'FRP', 'VISC'
    ]

def get_allen_color_dict():
    """
    Get Allen atlas colors formatted as dictionary of RGB arrays.
    :return:
    """
    # Get Allen atlas colors
    PATH_TO_ATLAS = r'C:\Users\bisi\.brainglobe\allen_mouse_bluebrain_barrels_10um_v1.0'

    with open(os.path.join(PATH_TO_ATLAS, 'structures.json')) as f:
        structures_dict = json.load(f)

    area_colors = {area['acronym']: np.array(area['rgb_triplet']) / 255 for area in structures_dict}
    return area_colors

def get_excluded_areas():
    """
    Retrieve a list of excluded area acronyms.
    :return: List of excluded area acronyms
    """
    excluded_areas = ["alv", "amc", "aco", "act", "arb", "ar", "bic", "bsc", "c", "cpd", "cbc", "cbp", "cbf", "AQ",
                      "epsc", "mfbc", "cett", "chpl", "cing", "cVIIIn", "fx", "stc", "cc", "fa", "ccb", "ee", "fp",
                      "ccs", "cst", "cm", "tspc", "cuf", "tspd", "dtd", "das", "dc", "df", "dhc", "lotd", "drt", "sctd",
                      "mfbse", "ec", "em", "eps", "VIIn", "fr", "fiber tracts", "fi", "fxs", "V4", "ccg", "gVIIn",
                      "hbc", "hc", "mfsbshy", "icp", "cic", "int", "lfbs", "ll", "lot", "lotg", "V4r", "mp",
                      "mfbsma", "mtg", "mtt", "mct", "mfb", "mfbs", "ml", "mlf", "mcp", "moV", "nst", "IIIn", "In",
                      "onl", "och", "IIn", "or", "opt", "fxpo", "pc", "pm", "py", "pyd", "root", "rust", "sV", "ts",
                      "sptV", "sm", "st", "SEZ", "scp", "dscp", "csc", "scwm", "sup", "tsp", "lfbst", "V3", "tb", "Vn",
                      "IVn", "uf", "Xn", "vhc", "sctv", "vtd", "VS", "vVIIIn", "VIIIn", "von",
                      'VL', 'I',
                      'nan'
                      ]
    return excluded_areas


def contains_layer(region):
    """Check if a region name contains a layer number excluding CA1, CA2, and CA3."""
    if region in ['CA1', 'CA2', 'CA3']:
        return False
    else:
        return bool(re.search(r'\d+[a-zA-Z]*', region))  # e.g., "6a", "6b"

def generalize_region(region):
    """Generalize region names based on predefined rules."""
    region_map = {
        "ACA": "ACA",
        "AD":"ATN",
        "AI": "AI",
        "AMd":"ATN",
        "AMv":"ATN",
        "AON":"OLF",
        "APN":"MB",
        "AUD": "AUD",
        "AV":"ATN",
        "BLAp":"BLA",
        "BST":"PAL",
        "CEA": "CEA",
        "CL": "ILM",
        "CM":"ILM",
        "DG": "DG",
        "Eth": "LAT",
        "EPd": "EP",
        "EPv": "EP",
        'HPF':'HPF',
        "HY":"HY",
        "IG":"HPF",
        "IGL":"LGN",
        "IntG":"LGN",
        "INC":"PAG",
        "LD":"ATN",
        "LGd": "LGN",
        "LGv": "LGN",
        "LH":"HA",
        "LS": "LS",
        "LT": "MY",
        "MD":"MED",
        "MH":"HA",
        "MS":"LGN",
        "MGm":"MGN",
        "MGv":"MGN",
        "MGd":"MGN",
        "MMd":"HY",
        "MMme":"HY",
        "NLL":"Pons",
        "NPC":"MB",
        "ORB": "ORB",
        "P":"Pons", # make sure it's only Pons
        "PAL": "PAL",
        "PIL":"ILM", # confirm, also, use DORpm vs DORsm ? PIL-PP in lit.
        "PIR":"OLF",
        "POL":"LAT",
        "POST":"HPF",
        "PoT":"VP",
        "PP":"ILM",# PIL-PP in lit., ILMN: intralaminar nuclei
        "PPN":"MB",
        "PR":"MED",
        "PRC":"PAG",
        "PRNr":"Pons",
        "RE":"MTN",
        "RPF":"MB",
        "RR":"MRN",
        "RSP": "RSP",
        "SAG":"MB",
        "SGN":"LAT",
        "SI":"PAL",
        "SMT":"MED",
        "STR": "STR",
        "SUM":"HY",
        "TEa": "TEa",
        "TRS":"PAL",
        "VPL": "VPL",
        "VPM": "VPM",
        "VM":"VPM",
        "Xi":"MTN"
    }
    for key in region_map:
        if region.startswith(key):
            return region_map[key]

    if region.startswith("LA"):
        return "LAT" if region.startswith("LAT") else "LA"

    if region.startswith("MY"): #medulla
        return "MY"

    if region.startswith("SC"):
        return "SCm" if region in ["SCdg", "SCdw", "SCig", "SCiw"] else "SCs"

    if region.startswith("SSp-bfd"):
        return "SSp-bfd"

    if region.startswith("VIS"):
        return "VISC" if region.startswith("VISC") else "VIS"

    return region  # Default: no change


def handle_ssp_bfd(region):
    """Special case: handle SSp-bfd barrels (e.g., "SSp-bfd-C4" -> "SSp-bfd")."""
    return re.sub(r'SSp-bfd-[A-Z]\d+', 'SSp-bfd', region) if "SSp-bfd" in region else region



def simplify_area(ccf_acronym, ccf_parent_acronym):
    """Decide and return the simplified area name."""
    base_region = ccf_acronym if not contains_layer(ccf_acronym) or ccf_acronym in ['CA1', 'CA2',
                                                                                    'CA3'] else ccf_parent_acronym
    return handle_ssp_bfd(generalize_region(base_region))


def create_area_custom_column(df):
    """
    Using helps, create a new column 'area_acronym_custom' based on 'ccf_acronym' and 'ccf_parent_acronym'.
    - If ccf_acronym contains a layer number, use ccf_parent_acronym unless the region is CA1, CA2, or CA3.
    - Simplifies visual areas (e.g., VISpm, VISa, VISal) to "VIS".
    - Simplifies auditory areas (e.g., AUDd, AUDpo, AUDp, AUDv) to "AUD".
    - Simplifies ORBv to "ORB".
    - Handles specific cases like SSp-bfd barrel indications (e.g., SSp-bfd-C4 -> SSp-bfd).

    :param df: A pandas DataFrame containing 'ccf_acronym' and 'ccf_parent_acronym' columns.
    :return: DataFrame with a new column 'area_acronym_custom'.
    """
    df['area_acronym_custom'] = df.apply(lambda row: simplify_area(row['ccf_acronym'], row['ccf_parent_acronym']), axis=1)
    return df

def extract_layer_info(ccf_acronym):
    """Extract and return layer information from a region name."""
    match = re.search(r'(\d+[a-zA-Z]*)', ccf_acronym)
    if match:
        layer = match.group(0)
        return "2/3" if layer == "2" else layer
    return None


def create_layer_number_column(df):
    """Create a column 'layer_number' that only extracts layer information."""
    df['layer_number'] = df['ccf_acronym'].apply(extract_layer_info)
    return df


def create_ccf_acronym_no_layer_column(df):
    """Create a column 'ccf_acronym_no_layer' that keeps the original ccf_acronym unless it contains a layer number, in which case it uses the parent acronym."""
    df['ccf_acronym_no_layer'] = df.apply(
        lambda row: handle_ssp_bfd(row['ccf_parent_acronym']) if contains_layer(row['ccf_acronym']) else row['ccf_acronym'], axis=1)
    return df

def get_custom_area_order():
    """
    Get the order of brain areas for plotting.
    """
    area_order = ['MOp', 'MOs', 'MOs-a', 'MOs-m', 'MOs-p', 'FRP', 'ACA', 'PL', 'ORB', 'AI',
                  'SSp-bfd', 'SSs', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-ll', 'SSp-tr', 'SSp-un',
                  'AUD', 'RSP',
                  'CLA', 'EP',
                  'CA1', 'CA2', 'CA3', 'DG', 'HPF',
                  'CP', 'DMS', 'DLS', 'TS', 'STR', 'ACB', 'VS', 'LS', 'SF', 'GPe', 'PAL', 'MS',
                  'VPL', 'VPM', 'LD', 'RT', 'PO', 'LGN', 'LP', 'ATN', 'LAT', 'MGN', 'MED', 'MTN', 'ILM', 'HA',
                  'SCs', 'SCm', 'MB', 'VTA', 'MRN', 'PAG', 'RN', 'SNr',
                  'Pons', 'MY',
                  'AON', 'OLF', 'PIR',
                  'BLA', 'LA', 'CEA','HY', 'ZI']
    return area_order

def get_custom_area_groups():
    """
    Get the custom area groups for plotting.
    """

    area_groups = {
        'Motor and frontal areas': ['MOp', 'MOs', 'MOs-a', 'MOs-m', 'MOs-p', 'FRP', 'ACA', 'PL', 'ORB', 'AI'],
        'Somatosensory areas': ['SSp-bfd', 'SSs', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-ll', 'SSp-tr', 'SSp-un'],
        'Auditory areas': ['AUD'],
        'Retrosplenial areas': ['RSP'],
        'Cortical subplate': ['CLA', 'EP'],
        'Hippocampus': ['CA1', 'CA2', 'CA3', 'DG', 'HPF'],
        'Striatum and pallidum': ['CP', 'DMS', 'DLS', 'TS', 'STR', 'VS', 'ACB', 'LS', 'SF', 'GPe', 'PAL', 'MS'],
        'Thalamus': ['VPL', 'VPM', 'LD', 'RT', 'PO', 'LGN', 'LP', 'ATN', 'LAT', 'MGN', 'MED', 'MTN', 'ILM', 'HA'],
        'Midbrain': ['SCs', 'SCm', 'MB', 'VTA', 'MRN', 'PAG', 'RN', 'SNr'],
        'Pons and medulla': ['Pons', 'MY'],
        'Olfactory areas': ['AON', 'OLF', 'PIR'],
        'Amygdala and hypothalamus': ['BLA', 'LA', 'CEA', 'HY', 'ZI']
    }
    return area_groups

def get_custom_area_groups_from_name():
    """
    Get custom area groups from a predefined dictionary based on area names.

    :return: Dictionary of area groups with area acronyms as keys
    """
    area_groups = get_custom_area_groups()
    area_groups_from_name = {area: group for group, areas in area_groups.items() for area in areas}
    return area_groups_from_name

def get_custom_area_groups_colors():
    """Get custom area group colors for plotting, here Allen colors."""
    area_group_colors = {
        'Motor and frontal areas': '#1f9d5a',
        'Somatosensory areas': '#188064',
        'Auditory areas': '#019399',
        'Retrosplenial areas': '#1aa698',
        'Cortical subplate': '#8ada87',
        'Hippocampus': '#7ed04b',
        'Striatum and pallidum': '#98d6f9',
        'Thalamus': '#ff7080',
        'Midbrain': '#ff64ff',
        'Pons and medulla': '#ffc395',
        'Olfactory areas': '#9ad2bd',
        'Amygdala and hypothalamus': '#f2483b'

    }
    return area_group_colors

def get_custom_area_color_per_group():
    """
    Using custom area group above, return a dictionary and list of single-area colors.

    """

    # Make cmap with as many colors as number of area groups
    group_color_palette = get_custom_area_groups_colors() # colors from Allen atlas
    area_groups = get_custom_area_groups() # potentially not all groups are present
    colors = [group_color_palette[i % len(group_color_palette)] for i in range(len(area_groups))]

    # Create a dictionary mapping each single area to its group color
    area_color_dict = {}
    for (group_name, areas), color in zip(area_groups.items(), colors):
        for area in areas:
            area_color_dict[area] = color

    # Make it also a list
    area_color_list = list(area_color_dict.values())
    return area_color_dict, area_color_list

def apply_target_region_filters(peth_table, area):
    """
    Apply specific area filters based on the area name.

    :param peth_area: Subset of PETH table for a specific area
    :param area: Specifically-named brain area
    :return: Filtered PETH table
    """
    specific_filters = {
        'wS1': ['SSp-bfd'],
        'wM1': ['MOp', 'MOs'],
        'wS2': ['SSs', 'SSp-bfd'],
        'wM2': ['MOp', 'MOs'],
        'mPFC': ['PL', 'ILA', 'ACA', 'ACAd', 'ACAv'],
        'tjM1': ['MOp', 'MOs'],
        'A1': ['AUD', 'AUDd', 'AUDp', 'AUDv', 'AUDpo'],
        'DLS': ['STRd', 'CP'],
        'SC': ['SC', 'SCs', 'SCiw', 'SCop', 'SCm', 'SCzo', 'SCsg'],
        'OFC': ['ORB', 'ORBm', 'ORBl', 'ORBvl'],
        'ALM': ['MOp', 'MOs'],
        'PPC': ['VISam', 'VISl', 'VISpm', 'VISrl', 'VISal', 'SSp-tr'],
    }

    if area in specific_filters.keys():
        # Keep only areas specified in filter and actually targeted e.g. SSp-bfd
        peth_area = peth_table[(peth_table['area_acronym_custom'].isin(specific_filters[area]))
                                & (peth_table['target_region'] == area)]
    else:
        print(f'{area} is not part of the target area dict. Skipping.')
        peth_area = None

    return peth_area

def create_bregma_centric_coords_from_ccf(df):
    """
    Convert CCF coordinates in BrainGlobe space into bregma-centric coordinates.
    i.e. from (0,0,0)=(A,S,R) anterior top right corner to (0,0,0)=bregma.
    Using IBL bregma estimate:
    https://docs.internationalbrainlab.org/_autosummary/iblatlas.atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM.html
    :param df: unit_table pd.DataFrame with columns 'ccf_ap', 'ccf_ml', 'ccf_dv', and 'mouse_id'.
    :return:
    """
    # Convert columns to numeric
    df[['ccf_ap', 'ccf_ml', 'ccf_dv']] = df[['ccf_ap', 'ccf_ml', 'ccf_dv']].astype(float)

    # TODO: update fcn after new NWBs
    new_nwb_mice = ['AB080', 'AB082', 'AB085', 'AB086', 'AB087', 'AB092', 'AB093', 'AB094', 'AB095',
                    'AB102', 'AB104', 'AB107',
                    'AB129', 'AB130']

    # Define conversion functions (all in um)
    #ml = (self.channels['x'] * 1e6) + 5739
    # Define conversion functions
    def func_to_ml(row):
        to_ml = lambda x: x - 5739
        if row['mouse_id'] in new_nwb_mice:
            return to_ml(row['ccf_ml'])
        else:
            return to_ml(row['ccf_dv']) #TODO: note, DV-ML inverted in NWB_Conversion -> update after new NWBs!

    #ap = (-self.channels['y'] * 1e6) + 5400
    def func_to_ap(row):
        to_ap = lambda x: -x + 5400 # AP positive is anterior relative to bregma
        if row['mouse_id'] in new_nwb_mice:
            return to_ap(row['ccf_ap'])
        else:
            return to_ap(row['ccf_ap'])
    #dv = (abs(self.channels['z'] * 1e6)) + 332
    def func_to_dv(row):
        to_dv = lambda x: x - 332
        if row['mouse_id'] in new_nwb_mice:
            return to_dv(row['ccf_dv'])
        else:
            return to_dv(row['ccf_ml']) #TODO: note, DV-ML inverted in NWB_Conversion -> update after new NWBs!

    # Apply conversions
    df['ap'] = df.apply(func_to_ap, axis=1)
    df['ml'] = df.apply(func_to_ml, axis=1)
    df['dv'] = df.apply(func_to_dv, axis=1)

    return df

def create_areas_subdivisions(df):
    """
    Divide large areas into smaller subdivisions for better visualization.
    :param df: unit_table pd.DataFrame with columns 'area_acronym_custom', 'ap', 'ml', 'dv'.
    :return:
    """
    parent_child_dict = {
        'CP': ['VS', 'DLS', 'DMS', 'TS'], # assignment order
        'MOs': ['MOs-a', 'MOs-m', 'MOs-p']
    }

    coord_boundaries = {
        'DMS': {'ap': (-1500, 6000), 'ml': (0, 2300), 'dv': (0, 7000)},
        'DLS': {'ap': (-1500, 6000), 'ml': (2300, 6000), 'dv': (0, 7000)},
        'TS': {'ap': (-6000, -1500), 'ml': (0, 6000), 'dv': (0, 7000)},
        'VS': {'ap': (0, 6000), 'ml': (0, 6000), 'dv': (4000, 7000)},
        'MOs-a': {'ap': (2500, 5000), 'ml': (-np.inf, np.inf), 'dv': (-np.inf, np.inf)},
        'MOs-m': {'ap': (1500, 2500), 'ml': (-np.inf, np.inf), 'dv': (-np.inf, np.inf)},
        'MOs-p': {'ap': (0, 1500), 'ml': (-np.inf, np.inf), 'dv': (-np.inf, np.inf)},
    }

    # Explicit assignment priority order — highest to lowest
    df['__assigned'] = False # temp col to track has been assigned

    for parent_area, subdivisions in parent_child_dict.items():
        # Filter rows belonging to the parent area once
        for sub_area in subdivisions:

            bounds = coord_boundaries[sub_area]
            unassigned_mask = ~df['__assigned']

            # For ventral striatum (VS), also include non-parent like STR, ACB (NAc)
            if sub_area == 'VS':
                parent_mask = (df['area_acronym_custom'].isin([parent_area,'STR','ACB'])
                               & unassigned_mask)
            else:
                parent_mask = (df['area_acronym_custom'] == parent_area) & unassigned_mask

            # Create masks for each axis considering infinite bounds
            ap_mask = df['ap'].between(*bounds['ap']) if np.isfinite(bounds['ap'][0]) and np.isfinite(bounds['ap'][1]) \
                else pd.Series(True, index=df.index)
            ml_mask = df['ml'].between(*bounds['ml']) if np.isfinite(bounds['ml'][0]) and np.isfinite(bounds['ml'][1]) \
                else pd.Series(True, index=df.index)
            dv_mask = df['dv'].between(*bounds['dv']) if np.isfinite(bounds['dv'][0]) and np.isfinite(bounds['dv'][1]) \
                else pd.Series(True, index=df.index)

            mask = parent_mask & ap_mask & ml_mask & dv_mask
            df.loc[mask, 'area_acronym_custom'] = sub_area
            df.loc[mask, '__assigned'] = True  # Mark as assigned
            print(f"{sub_area}: {mask.sum()} voxels assigned")

        # Check that subdivisions were applied correctly
        print('Unassigned', parent_area, len(df[df['area_acronym_custom'] == parent_area]))


    # Remove temp col
    df.drop(columns=['__assigned'], inplace=True)
    return df

def process_allen_labels(df, subdivide_areas=False):
    """
    Process the DataFrame to create custom area acronyms, layer numbers, and bregma-centric coordinates.
    :param df: unit_table pd.DataFrame from NWB files
    :param params: dictionary of parameters
    :return:
    """
    print('Processing CCF labels...')
    # Create custom area acronyms simplifying ccf areas acronyms
    df = create_area_custom_column(df)

    # Create layer number column
    df = create_layer_number_column(df)

    # Create a ccf_acronym_no_layer column, copy of ccf_acronym just without layer info
    df = create_ccf_acronym_no_layer_column(df)

    # Create bregma-centric coordinates, going from CCf (BrainGlobe) to bregma-centric coordinates using IBL bregma estimate
    df = create_bregma_centric_coords_from_ccf(df)

    # Create areas subdivisions for specific areas using custom boundaries
    if subdivide_areas:
        df = create_areas_subdivisions(df)

    return df