#!/usr/bin/env python3
"""
Convert .sps (SPECTRA) files to CSV format.
Simple command-line tool that asks for file path.
"""

import struct
import os
import csv
from pathlib import Path


def read_sps_file(filename):
    """
    Read an .sps file and extract wavelength and intensity data.

    Args:
        filename (str): Path to the .sps file

    Returns:
        tuple: (header_info, wavelengths, intensities)
    """

    with open(filename, 'rb') as f:
        # Read file header (typical SPS format)
        header = f.read(64)

        # Try to extract header information
        header_info = {}
        try:
            header_text = header.decode('ascii', errors='ignore').strip('\x00')
            if header_text:
                header_info['raw_header'] = header_text
        except:
            header_info['raw_header'] = 'Unable to decode header'

        # Read the rest of the file
        data = f.read()

        # Try different data formats
        num_floats = len(data) // 4

        if num_floats == 0:
            raise ValueError("File appears to be empty or corrupted")

        # Unpack floats
        floats = struct.unpack(f'{num_floats}f', data)

        # Try to determine if data is interleaved or sequential
        # Often format is: [wavelength1, wavelength2, ..., intensity1, intensity2, ...]
        midpoint = num_floats // 2

        wavelengths = list(floats[:midpoint])
        intensities = list(floats[midpoint:])

        # If lengths don't match, try interleaved format
        if len(wavelengths) != len(intensities):
            # Try interleaved: [w1, i1, w2, i2, ...]
            wavelengths = []
            intensities = []
            for i in range(0, len(floats), 2):
                if i + 1 < len(floats):
                    wavelengths.append(floats[i])
                    intensities.append(floats[i + 1])

        return header_info, wavelengths, intensities


def read_sps_as_text(filename):
    """
    Read .sps file as text.

    Args:
        filename (str): Path to the .sps file

    Returns:
        tuple: (header_info, wavelengths, intensities)
    """
    wavelengths = []
    intensities = []
    header_lines = []

    with open(filename, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Try to parse as two numbers (wavelength and intensity)
            parts = line.split()
            if len(parts) >= 2:
                try:
                    w = float(parts[0])
                    i = float(parts[1])
                    wavelengths.append(w)
                    intensities.append(i)
                    continue
                except ValueError:
                    pass

            # Try to parse as single number (might be sequential)
            try:
                val = float(line)
                # If we have alternating pattern, we'll detect later
                header_lines.append(val)
            except ValueError:
                header_lines.append(line)

    # If we couldn't find pairs, try to interpret as sequential data
    if not wavelengths and header_lines:
        numbers = [x for x in header_lines if isinstance(x, (int, float))]
        if numbers:
            midpoint = len(numbers) // 2
            wavelengths = numbers[:midpoint]
            intensities = numbers[midpoint:]

    header_info = {'header_lines': [str(x) for x in header_lines if not isinstance(x, (int, float))]}
    return header_info, wavelengths, intensities


def save_to_csv(output_file, wavelengths, intensities, header_info=None):
    """
    Save wavelength and intensity data to CSV file.
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header information as comments
        if header_info:
            for key, value in header_info.items():
                if isinstance(value, list) and value:
                    writer.writerow([f'# {key}:'])
                    for line in value[:10]:  # Limit header lines
                        if line:
                            writer.writerow([f'#   {line}'])
            writer.writerow([])

        # Write column headers
        writer.writerow(['Wavelength', 'Intensity'])

        # Write data
        for w, i in zip(wavelengths, intensities):
            writer.writerow([f'{w:.6f}', f'{i:.6f}'])


def convert_sps_to_csv(input_file, output_file=None, text_mode=False):
    """
    Convert a single SPS file to CSV.
    """
    print(f"\nProcessing: {input_file}")

    # Check if file exists
    if not os.path.exists(input_file):
        return False, f"Error: File not found - {input_file}"

    try:
        # Try to read the file
        if text_mode:
            print("  Using text mode...")
            header, wavelengths, intensities = read_sps_as_text(input_file)
        else:
            try:
                print("  Trying binary mode...")
                header, wavelengths, intensities = read_sps_file(input_file)
                print("  Binary mode successful")
            except (struct.error, UnicodeDecodeError, Exception) as e:
                print(f"  Binary mode failed: {str(e)}")
                print("  Falling back to text mode...")
                header, wavelengths, intensities = read_sps_as_text(input_file)

        # Check if we got data
        if not wavelengths or not intensities:
            return False, f"Error: No valid data found in {input_file}"

        if len(wavelengths) != len(intensities):
            print(f"  Warning: Wavelengths ({len(wavelengths)}) and intensities ({len(intensities)}) length mismatch")
            min_len = min(len(wavelengths), len(intensities))
            wavelengths = wavelengths[:min_len]
            intensities = intensities[:min_len]

        print(f"  Found {len(wavelengths)} data points")
        print(f"  Wavelength range: {min(wavelengths):.3f} - {max(wavelengths):.3f}")
        print(f"  Intensity range: {min(intensities):.6f} - {max(intensities):.6f}")

        # Determine output filename
        if not output_file:
            output_file = Path(input_file).with_suffix('.csv')

        # Save to CSV
        save_to_csv(output_file, wavelengths, intensities, header)
        print(f"  ✓ Saved to: {output_file}")

        return True, f"Successfully converted {os.path.basename(input_file)}"

    except Exception as e:
        return False, f"Error processing {input_file}: {str(e)}"


def main():
    print("=" * 60)
    print("SPS to CSV Converter")
    print("=" * 60)
    print("\nThis script converts .sps (SPECTRA) files to CSV format.")
    print("CSV files can be opened in Excel, Google Sheets, or any text editor.\n")

    while True:
        # Get file path from user
        print("\n" + "-" * 40)
        file_path = input("Enter the path to your .sps file (or 'quit' to exit): ").strip()

        if file_path.lower() in ['quit', 'q', 'exit']:
            print("\nGoodbye!")
            break

        # Remove quotes if present (common when copying paths)
        file_path = file_path.strip('"').strip("'")

        if not file_path:
            print("No file path entered. Please try again.")
            continue

        # Ask for conversion options
        print("\nConversion options:")
        print("  1. Auto-detect (recommended)")
        print("  2. Force text mode")

        mode_choice = input("Choose option (1/2) [default: 1]: ").strip()
        text_mode = (mode_choice == '2')

        # Ask for output file
        output_path = input("Enter output CSV path (press Enter for auto): ").strip()
        if not output_path:
            output_path = None
        else:
            output_path = output_path.strip('"').strip("'")

        # Convert the file
        success, message = convert_sps_to_csv(file_path, output_path, text_mode)
        print(f"\n{message}")

        if success:
            print("\n✓ Conversion complete!")
        else:
            print("\n✗ Conversion failed!")

        # Ask if user wants to convert another file
        again = input("\nConvert another file? (y/n): ").strip().lower()
        if again not in ['y', 'yes']:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nConversion cancelled by user. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        input("Press Enter to exit...")