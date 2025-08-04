
import csv
import os

def extract_core_data():
    """Extract only NUM;DATE;JACKPOT;N1;N2;N3;N4;N5;E1;E2 to txt file"""
    
    # Check if file exists
    csv_file = "euromillions.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    # Read CSV data and extract only required fields
    data = []
    
    with open(csv_file, 'r', encoding='utf-8-sig') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    if len(lines) < 2:
        print("CSV file has no data!")
        return
    
    # Get header from first line
    header = lines[0].split(';')
    print(f"Header columns: {header}")  # Debug
    
    # Process data rows
    for line in lines[1:]:
        if not line:
            continue
            
        parts = line.split(';')
        if len(parts) < 10:
            continue
            
        draw_data = {
            'NUM': parts[0].strip() if len(parts) > 0 else '',
            'DATE': parts[1].strip() if len(parts) > 1 else '',
            'JACKPOT': parts[2].strip() if len(parts) > 2 else '',
            'N1': parts[3].strip() if len(parts) > 3 else '',
            'N2': parts[4].strip() if len(parts) > 4 else '',
            'N3': parts[5].strip() if len(parts) > 5 else '',
            'N4': parts[6].strip() if len(parts) > 6 else '',
            'N5': parts[7].strip() if len(parts) > 7 else '',
            'E1': parts[8].strip() if len(parts) > 8 else '',
            'E2': parts[9].strip() if len(parts) > 9 else ''
        }
        
        if draw_data['NUM'] and draw_data['DATE']:
            data.append(draw_data)
    
    if not data:
        print("No data found in CSV!")
        return
    
    # Create CSV-like output with required fields only
    output_lines = []
    
    # Header
    output_lines.append("NUM;DATE;JACKPOT;N1;N2;N3;N4;N5;E1;E2")
    
    # Data rows
    for draw in data:
        row = f"{draw['NUM']};{draw['DATE']};{draw['JACKPOT']};{draw['N1']};{draw['N2']};{draw['N3']};{draw['N4']};{draw['N5']};{draw['E1']};{draw['E2']}"
        output_lines.append(row)
    
    # Save to file
    output_file = "euromillions_core_data.txt"
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(output_lines))
    
    print(f"  Données extraites sauvegardées dans: {output_file}")
    print(f"  Total des tirages traités: {len(data)}")
    
    # Display first few rows as preview
    print("\n  Aperçu des premières lignes:")
    print("-" * 80)
    for line in output_lines[:6]:
        print(line)
    print("-" * 80)

if __name__ == "__main__":
    extract_core_data()