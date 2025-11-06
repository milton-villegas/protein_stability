"""
Protein Stability DOE
Automates the preparation of buffer conditions for protein stability experiments.
It reads buffer recipes from a CSV file

Milton F. Villegas - v1.0.1
API Level: 2.20
"""

from opentrons import protocol_api
from opentrons.protocol_api import ParameterContext
import math

# PROTOCOL METADATA

metadata = {
    'protocolName': 'Protein Stability DOE v1.0.1',
    'author': 'Milton F. Villegas',
    'description': 'CSV-driven automated buffer preparation (supports up to 4 plates)',
    'apiLevel': '2.20'
}

# User parameters

def add_parameters(parameters: ParameterContext):
    """Define user parameters that appear in the Opentrons App interface"""

    # CSV file containing buffer conditions
    parameters.add_csv_file(
        variable_name="buffer_csv",
        display_name="Buffer Conditions CSV",
        description="Upload CSV with reagent names (row 1) and volumes (subsequent rows)."
    )

    # Transfer volume for 96→384 step
    parameters.add_int(
        variable_name="transfer_volume",
        display_name="Transfer Volume (µL)",
        description="Volume to transfer from each 96-well column to 384-well plate",
        default=50,
        minimum=1,
        maximum=112
    )

    # Enable/disable 96→384 transfer
    parameters.add_bool(
        variable_name="do_transfer_to_384",
        display_name="Transfer to 384 Plate?",
        description="Enable to perform 96→384 transfer after buffer preparation.",
        default=True
    )

# Viscous reagent profiles

"""
Viscous reagents with optimized flow rates and air gaps.
Protocol auto-detects these by name and adjusts pipetting.
Flow rates in µL/sec, air gaps in µL.
"""

VISCOUS_REAGENTS = {
    # High viscosity reagents
    'glycerol':         {'asp_rate': 15, 'disp_rate': 30, 'air_gap': 5},
    'peg 8000':         {'asp_rate': 15, 'disp_rate': 25, 'air_gap': 5},

    # Medium-high viscosity
    'eg':               {'asp_rate': 15, 'disp_rate': 30, 'air_gap': 3},
    'ethylene glycol':  {'asp_rate': 15, 'disp_rate': 30, 'air_gap': 3},
    'peg':              {'asp_rate': 20, 'disp_rate': 35, 'air_gap': 4},
    'peg 400':          {'asp_rate': 20, 'disp_rate': 35, 'air_gap': 4},

    # Medium viscosity
    'trehalose':        {'asp_rate': 25, 'disp_rate': 40, 'air_gap': 2},
    'sucrose':          {'asp_rate': 25, 'disp_rate': 40, 'air_gap': 2},
    'dmso':             {'asp_rate': 30, 'disp_rate': 50, 'air_gap': 1},

    # Surfactants (specifically used to prevent dripping)
    'tween':            {'asp_rate': 10, 'disp_rate': 20, 'air_gap': 5},
    'tween 20':         {'asp_rate': 10, 'disp_rate': 20, 'air_gap': 5},
    'triton':           {'asp_rate': 12, 'disp_rate': 25, 'air_gap': 4}
}

# Default rates for water-like reagents
DEFAULT_RATES = {'asp_rate': 40, 'disp_rate': 80, 'air_gap': 0}


def get_reagent_profile(reagent_name):
    """Get pipetting parameters based on reagent viscosity"""
    reagent_lower = reagent_name.lower().strip()

    # Exact match
    if reagent_lower in VISCOUS_REAGENTS:
        return VISCOUS_REAGENTS[reagent_lower]

    # Partial match (for words like "PEG 8000" matches "peg 8000")
    for viscous_name, profile in VISCOUS_REAGENTS.items():
        if viscous_name in reagent_lower:
            return profile

    # Default for water-like reagents
    return DEFAULT_RATES

# CSV PARSING

def parse_csv_opentrons(csv_param, protocol):
    """Parse CSV for Opentrons API 2.20 - tries multiple methods"""
    import csv as csv_module

    rows = None
    protocol.comment(f"CSV type: {type(csv_param).__name__}")

    # Method 1: .rows attribute (newer API)
    if hasattr(csv_param, "rows") and csv_param.rows:
        rows = csv_param.rows
        protocol.comment("Using .rows attribute")

    # Method 2: .data attribute
    elif hasattr(csv_param, "data") and csv_param.data:
        rows = csv_param.data
        protocol.comment("Using .data attribute")

    # Method 3: .parse_as_csv() method
    elif hasattr(csv_param, "parse_as_csv"):
        rows = csv_param.parse_as_csv()
        protocol.comment("Using .parse_as_csv() method")

    # Method 4: Direct string content
    elif isinstance(csv_param, str):
        rows = list(csv_module.reader(csv_param.strip().splitlines()))
        protocol.comment("Parsed as string")

    # Method 5: .contents or .content attribute
    elif hasattr(csv_param, "contents"):
        csv_text = csv_param.contents
        rows = list(csv_module.reader(csv_text.strip().splitlines()))
        protocol.comment("Using .contents attribute")

    elif hasattr(csv_param, "content"):
        csv_text = csv_param.content
        rows = list(csv_module.reader(csv_text.strip().splitlines()))
        protocol.comment("Using .content attribute")

    # Method 6: String conversion as last resort
    else:
        try:
            csv_text = str(csv_param)
            rows = list(csv_module.reader(csv_text.strip().splitlines()))
            protocol.comment("Using str() conversion")
        except Exception as e:
            protocol.comment(f"ERROR: Could not parse CSV - {str(e)}")
            raise ValueError("Could not extract data from CSV parameter")

    if not rows:
        raise ValueError("CSV parsing failed - no data extracted")

    protocol.comment(f"CSV has {len(rows)} total rows")

    # Validate structure
    if len(rows) < 2:
        raise ValueError(f"CSV must have ≥2 rows (headers + data). Found: {len(rows)}")

    # Parse headers (row 1)
    headers = [str(cell).strip() for cell in rows[0]]
    if not all(headers):
        raise ValueError("Header row must not have empty cells")

    protocol.comment(f"Found {len(headers)} reagents: {', '.join(headers[:3])}...")

    # Parse data rows (rows 2+)
    data_rows = []
    for i, row in enumerate(rows[1:], start=2):
        # Skip empty rows
        if not row or not any(str(cell).strip() for cell in row):
            continue

        # Pad row with zeros if shorter than headers
        if len(row) < len(headers):
            row += ['0'] * (len(headers) - len(row))

        # Convert all cells to floats
        parsed_row = []
        for j, cell in enumerate(row[:len(headers)]):
            try:
                val = float(str(cell).strip() or 0)
            except ValueError:
                raise ValueError(
                    f"Row {i}, column {j+1} ('{headers[j]}'): "
                    f"cannot convert '{cell}' to number"
                )
            parsed_row.append(val)

        data_rows.append(parsed_row)

    if not data_rows:
        raise ValueError("CSV has no valid data rows after headers")

    protocol.comment(f"Parsed {len(data_rows)} buffer conditions")
    return headers, data_rows


def generate_well_indices(n):
    """
    Generate well names in column-major order for 96-well plate.
    """
    wells = []
    for col in range(1, 13):  # Columns 1-12
        for row in range(8):  # Rows A-H
            wells.append(f"{chr(65 + row)}{col}")
            if len(wells) >= n:
                return wells
    return wells


def map_reagents_to_24well_reservoir(reagent_names, protocol):
    """Map reagents to 24-well reservoir positions"""
    if len(reagent_names) > 24:
        protocol.comment(
            f"ERROR: CSV has {len(reagent_names)} reagents "
            f"but reservoir only holds 24"
        )
        return None

    mapping = {}
    row_labels = ['A', 'B', 'C', 'D']
    idx = 0

    # Fill column by column (A1, B1, C1, D1, then A2, B2...)
    for col in range(1, 7):  # Columns 1-6
        for row in row_labels:  # Rows A-D
            if idx >= len(reagent_names):
                break
            well_name = f"{row}{col}"
            mapping[reagent_names[idx]] = well_name
            idx += 1
        if idx >= len(reagent_names):
            break

    return mapping


def calculate_plates_needed(num_conditions):
    """Calculate 96-well plates needed (max 4 plates)"""
    # Calculate columns needed (8 wells per column)
    columns_needed = math.ceil(num_conditions / 8)

    if columns_needed <= 12:
        return 1
    elif columns_needed <= 24:
        return 2
    elif columns_needed <= 36:
        return 3
    elif columns_needed <= 48:
        return 4
    else:
        max_conditions = 48 * 8  # 384
        raise ValueError(
            f"Too many conditions ({num_conditions}). "
            f"Maximum is {max_conditions} (4 plates × 12 columns × 8 wells). "
            f"You need {columns_needed} columns."
        )

# MAIN PROTOCOL

def run(protocol: protocol_api.ProtocolContext):
    """Main protocol: parse CSV, load plates, prepare buffers, transfer to 384"""

    # HEADER

    protocol.comment("PROTEIN STABILITY DOE v1.0")
    protocol.comment("Automated Buffer Preparation")

    # LOAD PARAMETERS

    protocol.comment("Loading parameters...")
    buffer_csv = protocol.params.buffer_csv
    # Transfer parameters
    start_384_col = 1  # Always start from column 1
    user_transfer_vol = protocol.params.transfer_volume
    do_transfer = protocol.params.do_transfer_to_384

    # PARSE CSV FILE

    protocol.comment("PARSING CSV FILE")

    try:
        headers, data_rows = parse_csv_opentrons(buffer_csv, protocol)
    except Exception as e:
        protocol.comment(f"FATAL ERROR parsing CSV: {str(e)}")
        protocol.comment("Please check your CSV file format and try again.")
        return


    # CALCULATE PLATES NEEDED

    num_conditions = len(data_rows)
    try:
        num_plates = calculate_plates_needed(num_conditions)
        columns_needed = math.ceil(num_conditions / 8)

        protocol.comment("AUTOMATIC PLATE CALCULATION")
        protocol.comment(f"  Conditions in CSV: {num_conditions}")
        protocol.comment(f"  Columns needed: {columns_needed} (8 wells/column)")
        protocol.comment(f"  Plates required: {num_plates}")

    except ValueError as e:
        protocol.comment(f"FATAL ERROR: {str(e)}")
        return

    # VALIDATE TRANSFER VOLUME

    if user_transfer_vol > 112:
        protocol.comment(f"Warning: Transfer volume ({user_transfer_vol}µL) exceeds")
        protocol.comment(f"   384-well capacity (112µL). Capping at 112µL.")
        user_transfer_vol = 112

    # PROTOCOL SETTINGS

    # Aspiration/dispense clearances from well bottom
    ASP_CLEAR = 0.5          # mm above bottom when aspirating
    DISP_CLEAR = 1.0         # mm above bottom when dispensing
    SHALLOW_OFFSET = -1      # mm from top for shallow dispenses
    PRIME_VOL = 5            # µL to prime tip before dispensing
    BLOW_OUT_DEST = True     # Blow out after each dispense

    # Mixing optimization (for 96→384 transfer)
    MIX_REPETITIONS = 3      # Number of mix cycles
    MIX_FLOW_RATE = 150      # µL/sec for faster mixing

    # LOAD LABWARE

    protocol.comment("LOADING LABWARE")

    # 96-well plates
    plates_96 = []
    if num_plates >= 1:
        plates_96.append(
            protocol.load_labware(
                'greiner_96_well_u_bottom_323ul', 1,
                '96-well Plate #1'
            )
        )
        protocol.comment("  Slot 1: 96-well Plate #1")

    if num_plates >= 2:
        plates_96.append(
            protocol.load_labware(
                'greiner_96_well_u_bottom_323ul', 3,
                '96-well Plate #2'
            )
        )
        protocol.comment("  Slot 3: 96-well Plate #2")

    if num_plates >= 3:
        plates_96.append(
            protocol.load_labware(
                'greiner_96_well_u_bottom_323ul', 4,
                '96-well Plate #3'
            )
        )
        protocol.comment("  Slot 4: 96-well Plate #3")

    if num_plates >= 4:
        plates_96.append(
            protocol.load_labware(
                'greiner_96_well_u_bottom_323ul', 6,
                '96-well Plate #4'
            )
        )
        protocol.comment("  Slot 6: 96-well Plate #4")

    # 384-well plate
    plate384 = protocol.load_labware(
        'corning_384_wellplate_112ul_flat', 2,
        '384-well Destination'
    )
    protocol.comment("  Slot 2: 384-well Plate")

    # Reagent reservoir
    reservoir = protocol.load_labware(
        'cytiva_24_reservoir_10ml', 5,
        '24-well Reservoir'
    )
    protocol.comment("  Slot 5: 24-well Reservoir")

    # Tip racks - Single channel
    tiprack_sc_1 = protocol.load_labware('opentrons_96_tiprack_300ul', 9,
                                         'Single channel tips')
    protocol.comment("  Slot 9: Single channel tips")
    
    # Tip racks - Multichannel

    tiprack_mc_list = []
    
    if num_plates >= 1:
        tiprack_mc_1 = protocol.load_labware('opentrons_96_tiprack_300ul', 8,
                                             'Multichannel tips (rack 1)')
        tiprack_mc_list.append(tiprack_mc_1)
        protocol.comment("  Slot 8: Multichannel tips (rack 1)")
    
    if num_plates >= 2:
        tiprack_mc_2 = protocol.load_labware('opentrons_96_tiprack_300ul', 7,
                                             'Multichannel tips (rack 2)')
        tiprack_mc_list.append(tiprack_mc_2)
        protocol.comment("  Slot 7: Multichannel tips (rack 2)")
    
    if num_plates >= 3:
        tiprack_mc_3 = protocol.load_labware('opentrons_96_tiprack_300ul', 11,
                                             'Multichannel tips (rack 3)')
        tiprack_mc_list.append(tiprack_mc_3)
        protocol.comment("  Slot 11: Multichannel tips (rack 3)")
    
    if num_plates >= 4:
        tiprack_mc_4 = protocol.load_labware('opentrons_96_tiprack_300ul', 10,
                                             'Multichannel tips (rack 4)')
        tiprack_mc_list.append(tiprack_mc_4)
        protocol.comment("  Slot 10: Multichannel tips (rack 4)")
    
    protocol.comment("")

    # LOAD PIPETTES

    protocol.comment("LOADING PIPETTES")

    # Single channel
    p300 = protocol.load_instrument(
        'p300_single',
        mount='right',
        tip_racks=[tiprack_sc_1]
    )
    # Set clearance heights to avoid hitting bottom/top
    p300.well_bottom_clearance.aspirate = ASP_CLEAR
    p300.well_bottom_clearance.dispense = DISP_CLEAR
    protocol.comment("  Right mount: P300 Single")

    # Multichannel
    m300 = protocol.load_instrument(
        'p300_multi',
        mount='left',
        tip_racks=tiprack_mc_list
    )
    m300.flow_rate.aspirate = 30
    m300.flow_rate.dispense = 30
    m300.well_bottom_clearance.aspirate = 0.5
    protocol.comment("  Left mount: P300 Multichannel")

    # MAP REAGENTS TO RESERVOIR

    reagent_mapping = map_reagents_to_24well_reservoir(headers, protocol)
    if not reagent_mapping:
        return

    # Calculate total volumes needed (sum across all wells)
    volume_totals = {reagent: 0 for reagent in headers}
    for row in data_rows:
        for i, vol in enumerate(row):
            volume_totals[headers[i]] += vol

    # Display reservoir setup
    protocol.comment("RESERVOIR SETUP")
    protocol.comment("  Please load reagents as follows:")

    for reagent, well in reagent_mapping.items():
        profile = get_reagent_profile(reagent)
        total_vol = volume_totals[reagent] / 1000  # Convert to mL
        viscous_flag = " [VISCOUS]" if profile != DEFAULT_RATES else ""
        protocol.comment(f"  {well}: {reagent:<20} ({total_vol:>5.2f} mL){viscous_flag}")

    protocol.comment("  Note: [VISCOUS] reagents use adjusted flow rates")

    # CREATE WELL DATA STRUCTURE

    # Build a list of all wells to fill with their reagent volumes
    all_well_data = []

    for plate_idx, plate in enumerate(plates_96):
        # Get the slice of data rows for this plate
        start_idx = plate_idx * 96
        end_idx = min(start_idx + 96, len(data_rows))
        plate_rows = data_rows[start_idx:end_idx]

        if not plate_rows:
            continue


        well_indices = generate_well_indices(len(plate_rows))


        for row_vals, well_idx in zip(plate_rows, well_indices):
            well_data = {
                'well-index': well_idx,
                'plate': plate,
                'plate_idx': plate_idx
            }

            for j, reagent in enumerate(headers):
                well_data[reagent] = row_vals[j]

            all_well_data.append(well_data)

    protocol.comment(f"Preparing {len(all_well_data)} wells across {len(plates_96)} plate(s)")

    # PHASE 1: BUFFER PREPARATION

    protocol.comment("PHASE 1: BUFFER PREPARATION")

    # Process each reagent one at a time
    for reagent_idx, reagent in enumerate(headers):
        protocol.comment(f"  Reagent {reagent_idx + 1}/{len(headers)}: {reagent}")

        # Get source well from reservoir
        source_well = reagent_mapping[reagent]
        source = reservoir[source_well]

        # Get viscosity profile for this reagent
        profile = get_reagent_profile(reagent)

        # Set flow rates based on viscosity
        p300.flow_rate.aspirate = profile['asp_rate']
        p300.flow_rate.dispense = profile['disp_rate']

        protocol.comment(f"  Source: {source_well}")
        if profile != DEFAULT_RATES:
            protocol.comment(
                f"  Flow rates: {profile['asp_rate']}/{profile['disp_rate']} µL/s "
                f"(aspirate/dispense)"
            )
            protocol.comment(f"  Air gap: {profile['air_gap']} µL")

        # Pick up tip
        p300.pick_up_tip()

        # Prime tip to remove air bubbles and ensure accurate dispensing
        p300.aspirate(PRIME_VOL, source)
        p300.dispense(PRIME_VOL, source.top())
        p300.blow_out(source.top())

        # Dispense reagent to all wells that need it
        dispense_count = 0
        for well_data in all_well_data:
            vol = float(well_data[reagent])

            # Skip wells that don't need this reagent
            if vol <= 0:
                continue

            # Get destination well
            dest_plate = well_data['plate']
            dest_well = dest_plate[well_data['well-index']]

            # Aspirate from reservoir
            p300.aspirate(vol, source)

            # Add air gap if needed (for viscous reagents)
            if profile['air_gap'] > 0:
                p300.air_gap(profile['air_gap'])

            # Dispense to well (shallow, near top)
            p300.dispense(
                vol + profile['air_gap'],
                dest_well.top(SHALLOW_OFFSET)
            )

            # Blow out to ensure complete dispensing
            if BLOW_OUT_DEST:
                p300.blow_out(dest_well.top())

            dispense_count += 1

        protocol.comment(f"  Dispensed to {dispense_count} wells")

        # Drop tip
        p300.drop_tip(home_after=False)

    protocol.comment("Buffer preparation complete!")

    # PHASE 2: 96 → 384 TRANSFER

    if not do_transfer:
        # Skip transfer phase if disabled
        protocol.comment("SKIPPING 96→384 TRANSFER")
        protocol.comment("PROTOCOL COMPLETE")
        protocol.comment(f"   Prepared {len(all_well_data)} buffer conditions")
        return

    protocol.comment("PHASE 2: 96-WELL → 384-WELL TRANSFER")
    protocol.comment(f"  Transfer volume: {user_transfer_vol} µL")
    protocol.comment(f"  Starting at 384 column: 1 (auto-fills entire plate)")

    # Create interleaved well list for 384 plate

    def create_384_interleaved_wells(plate384, start_col):
        """
        Create interleaved destination list for 384 plate.
        """
        interleaved = []
        for col in range(start_col, 25):  # Columns 1-24
            col_wells = []
            # Get all 16 rows (A-P) for this column
            for row in range(16):
                row_letter = chr(ord('A') + row)
                col_wells.append(plate384[f"{row_letter}{col}"])

            # Take only rows A and B, alternating
            interleaved.append(col_wells[0])  # Row A
            interleaved.append(col_wells[1])  # Row B

        return interleaved

    wells_384_interleaved = create_384_interleaved_wells(plate384, start_384_col)
    transfer_idx = 0

    # Transfer each 96-well column to 384 plate

    for plate_idx, plate_96 in enumerate(plates_96):
        # Get wells for this plate
        plate_wells = [w for w in all_well_data if w['plate_idx'] == plate_idx]
        if not plate_wells:
            continue

        # Determine which columns are used
        used_cols = sorted(set(int(w['well-index'][1:]) for w in plate_wells))

        protocol.comment(f"  96-well Plate #{plate_idx + 1}: {len(used_cols)} columns")

        # Transfer each column
        for col_idx, col in enumerate(used_cols):
            # Check if we've run out of space in 384 plate
            if transfer_idx >= len(wells_384_interleaved):
                protocol.comment("  WARNING: Reached end of 384 plate!")
                break

            # Source: top well of column in 96-plate (multichannel uses A row)
            source_96 = plate_96[f'A{col}']

            # Destination: next well in interleaved 384 list
            dest_384 = wells_384_interleaved[transfer_idx]

            protocol.comment(f"  Column {col_idx + 1}: {source_96} → {dest_384}")

            # Pick up tips with multichannel
            m300.pick_up_tip()

            # Mix before transfer
            # Use 80% of transfer volume (max 80 µL to avoid overflow)
            mix_volume = min(user_transfer_vol * 0.8, 80)

            # Increase flow rate for faster mixing
            original_flow_rate = m300.flow_rate.aspirate
            m300.flow_rate.aspirate = MIX_FLOW_RATE
            m300.flow_rate.dispense = MIX_FLOW_RATE

            # Mix
            m300.mix(MIX_REPETITIONS, mix_volume, source_96.bottom(1))

            # Normal flow rate for transfer
            m300.flow_rate.aspirate = original_flow_rate
            m300.flow_rate.dispense = original_flow_rate

            # Transfer to 384 plate

            m300.transfer(
                user_transfer_vol,
                source_96.bottom(0.5),      # Aspirate from near bottom
                dest_384.top(-1),           # Dispense near top
                new_tip='never',            # Already have tip
                air_gap=1,                  # Small air gap to prevent dripping
                blow_out=True,              # Blow out to ensure complete transfer
                blowout_location='destination well'
            )

            # Drop tip (skip homing for speed - saves ~2s per tip)
            m300.drop_tip(home_after=False)

            transfer_idx += 1

    protocol.comment("Transfer complete!")

    # PROTOCOL SUMMARY

    protocol.comment("PROTOCOL COMPLETE!")
    protocol.comment("Summary:")
    protocol.comment(f"  • Buffer conditions: {num_conditions}")
    protocol.comment(f"  • 96-well plates used: {num_plates}")
    protocol.comment(f"  • Total wells prepared: {len(all_well_data)}")
    protocol.comment(f"  • Columns transferred to 384: {transfer_idx}")

    # Calculate which 384 columns were filled
    cols_filled = (transfer_idx + 1) // 2
    end_col = cols_filled
    protocol.comment(f"  • 384 plate columns filled: 1 to {end_col}")
