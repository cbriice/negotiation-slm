#!/usr/bin/env python3
"""
Script to clean problematic Unicode characters from files.
Replaces confusing Unicode characters with their ASCII equivalents.
"""
import sys
import argparse
from pathlib import Path
import re

# Mapping of problematic Unicode characters to their ASCII replacements
UNICODE_REPLACEMENTS = {
    # Dashes and hyphens
    '\u2013': '-',     # En dash (–) → hyphen
    '\u2014': '--',    # Em dash (—) → double hyphen
    '\u2010': '-',     # Hyphen (‐) → ASCII hyphen
    '\u2011': '-',     # Non-breaking hyphen → ASCII hyphen
    '\u2012': '-',     # Figure dash → ASCII hyphen
    '\u2015': '--',    # Horizontal bar → double hyphen
    '\u2212': '-',     # Minus sign → ASCII hyphen

    # Quotes
    '\u201c': '"',     # Left double quotation mark (")
    '\u201d': '"',     # Right double quotation mark (")
    '\u201e': '"',     # Double low quotation mark („)
    '\u201f': '"',     # Double high quotation mark (‟)
    '\u2018': "'",     # Left single quotation mark (')
    '\u2019': "'",     # Right single quotation mark (')
    '\u201a': "'",     # Single low quotation mark (‚)
    '\u201b': "'",     # Single high quotation mark (‛)
    '\u00ab': '"',     # Left guillemet («)
    '\u00bb': '"',     # Right guillemet (»)

    # Spaces
    '\u00a0': ' ',     # Non-breaking space
    '\u2000': ' ',     # En quad
    '\u2001': ' ',     # Em quad
    '\u2002': ' ',     # En space
    '\u2003': ' ',     # Em space
    '\u2004': ' ',     # Three-per-em space
    '\u2005': ' ',     # Four-per-em space
    '\u2006': ' ',     # Six-per-em space
    '\u2007': ' ',     # Figure space
    '\u2008': ' ',     # Punctuation space
    '\u2009': ' ',     # Thin space
    '\u200a': ' ',     # Hair space
    '\u202f': ' ',     # Narrow no-break space
    '\u205f': ' ',     # Medium mathematical space
    '\u3000': ' ',     # Ideographic space

    # Other punctuation
    '\u2026': '...',   # Ellipsis (…)
    '\u2022': '*',     # Bullet (•)
    '\u2023': '>',     # Triangular bullet (‣)
    '\u2043': '-',     # Hyphen bullet (⁃)
    '\u2219': '*',     # Bullet operator (∙)
    '\u25e6': 'o',     # White bullet (◦)
    '\u2024': '.',     # One dot leader (․)
    '\u2025': '..',    # Two dot leader (‥)
    '\u2027': '-',     # Hyphenation point (‧)

    # Apostrophes and similar
    '\u02bc': "'",     # Modifier letter apostrophe (ʼ)
    '\u055a': "'",     # Armenian apostrophe (՚)
    '\u07f4': "'",     # NKo high tone apostrophe (ߴ)
    '\u07f5': "'",     # NKo low tone apostrophe (ߵ)
    '\u2032': "'",     # Prime (′)
    '\u2033': '"',     # Double prime (″)
    '\u2034': "'''",   # Triple prime (‴)
    '\u2057': "''''",  # Quadruple prime (⁗)

    # Arrows that might be used as hyphens/dashes
    '\u2192': '->',    # Rightwards arrow (→)
    '\u2190': '<-',    # Leftwards arrow (←)
    '\u2194': '<->',   # Left-right arrow (↔)

    # Mathematical symbols that might appear
    '\u00d7': 'x',     # Multiplication sign (×)
    '\u00f7': '/',     # Division sign (÷)
    '\u2044': '/',     # Fraction slash (⁄)
    '\u2215': '/',     # Division slash (∕)

    # Other common problematic characters
    '\ufeff': '',      # Zero width no-break space (BOM)
    '\u200b': '',      # Zero width space
    '\u200c': '',      # Zero width non-joiner
    '\u200d': '',      # Zero width joiner
    '\u200e': '',      # Left-to-right mark
    '\u200f': '',      # Right-to-left mark
    '\u202a': '',      # Left-to-right embedding
    '\u202b': '',      # Right-to-left embedding
    '\u202c': '',      # Pop directional formatting
    '\u202d': '',      # Left-to-right override
    '\u202e': '',      # Right-to-left override
    '\u2060': '',      # Word joiner
    '\u2061': '',      # Function application
    '\u2062': '',      # Invisible times
    '\u2063': '',      # Invisible separator
    '\u2064': '',      # Invisible plus
    '\u206a': '',      # Inhibit symmetric swapping
    '\u206b': '',      # Activate symmetric swapping
    '\u206c': '',      # Inhibit Arabic form shaping
    '\u206d': '',      # Activate Arabic form shaping
    '\u206e': '',      # National digit shapes
    '\u206f': '',      # Nominal digit shapes
    '\ufffe': '',      # Noncharacter
    '\uffff': '',      # Noncharacter
}

def clean_file(file_path, output_path=None, backup=True):
    """
    Clean problematic Unicode characters from a file.

    Args:
        file_path: Path to the file to clean
        output_path: Optional path for output (defaults to overwriting original)
        backup: Whether to create a backup of the original file

    Returns:
        Number of replacements made
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File '{file_path}' not found")
        return -1

    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encodings if UTF-8 fails
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"Successfully read file using {encoding} encoding")
                break
            except:
                continue
        else:
            print(f"Error: Could not decode file '{file_path}'")
            return -1

    original_content = content
    replacement_count = 0
    replacements_made = {}

    # FIRST: Handle inch markers (must be done before general quote replacements)
    # Pattern matches numbers followed by double quotes (inches)
    # Includes both ASCII and Unicode double quote variants
    inch_pattern = r'(\d+(?:\.\d+)?)\s*["\u201c\u201d\u201e\u201f\u2033"]'
    inch_matches = re.findall(inch_pattern, content)
    if inch_matches:
        # Replace inch markers with " inches" to avoid JSON/dict formatting issues
        content_modified = re.sub(inch_pattern, r'\1 inches', content)
        inch_replacement_count = len(re.findall(inch_pattern, content))
        replacement_count += inch_replacement_count
        replacements_made['inch markers'] = inch_replacement_count
        content = content_modified
        print(f"Replaced {inch_replacement_count} inch markers (e.g., 32\" → 32 inches)")

    # Replace problematic characters
    for unicode_char, ascii_replacement in UNICODE_REPLACEMENTS.items():
        if unicode_char in content:
            count = content.count(unicode_char)
            replacement_count += count
            replacements_made[unicode_char] = count
            content = content.replace(unicode_char, ascii_replacement)

    # Create backup if requested and changes were made
    if backup and replacement_count > 0 and output_path is None:
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"Backup created: {backup_path}")

    # Write cleaned content
    if replacement_count > 0:
        output = output_path if output_path else file_path
        with open(output, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Cleaned file: {output}")
        print(f"Total replacements: {replacement_count}")

        # Show what was replaced
        if replacements_made:
            print("\nReplacements made:")
            for char, count in sorted(replacements_made.items(), key=lambda x: x[1], reverse=True):
                # Handle inch markers specially
                if char == 'inch markers':
                    print(f"  Inch markers (e.g., 32\" → 32 inches): {count} occurrences")
                else:
                    char_name = f"U+{ord(char):04X}"
                    replacement = UNICODE_REPLACEMENTS[char]
                    if replacement:
                        print(f"  {char_name} ({char}) → '{replacement}': {count} occurrences")
                    else:
                        print(f"  {char_name} (removed): {count} occurrences")
    else:
        print(f"No problematic Unicode characters found in {file_path}")

    return replacement_count

def main():
    parser = argparse.ArgumentParser(description='Clean problematic Unicode characters from files')
    parser.add_argument('files', nargs='+', help='File(s) to clean')
    parser.add_argument('-o', '--output', help='Output file (only works with single input file)')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be replaced without modifying files')

    args = parser.parse_args()

    if args.output and len(args.files) > 1:
        print("Error: --output can only be used with a single input file")
        sys.exit(1)

    total_replacements = 0
    for file_path in args.files:
        print(f"\nProcessing: {file_path}")
        print("-" * 40)

        if args.dry_run:
            # Dry run - just show what would be replaced
            file_path = Path(file_path)
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    found = {}

                    # Check for inch markers
                    inch_pattern = r'(\d+(?:\.\d+)?)\s*["\u201c\u201d\u201e\u201f\u2033"]'
                    inch_matches = re.findall(inch_pattern, content)
                    if inch_matches:
                        inch_count = len(inch_matches)
                        found['inch markers'] = inch_count

                    # Check for other Unicode characters
                    for unicode_char, ascii_replacement in UNICODE_REPLACEMENTS.items():
                        if unicode_char in content:
                            found[unicode_char] = content.count(unicode_char)

                    if found:
                        print("Would replace:")
                        # Handle inch markers first if present
                        if 'inch markers' in found:
                            print(f"  Inch markers (e.g., 32\" → 32 inches): {found['inch markers']} occurrences")
                            del found['inch markers']

                        # Handle other replacements
                        for char, count in sorted(found.items(), key=lambda x: x[1], reverse=True):
                            char_name = f"U+{ord(char):04X}"
                            replacement = UNICODE_REPLACEMENTS[char]
                            if replacement:
                                print(f"  {char_name} ({char}) → '{replacement}': {count} occurrences")
                            else:
                                print(f"  {char_name} (would be removed): {count} occurrences")
                    else:
                        print("No problematic Unicode characters found")
                except Exception as e:
                    print(f"Error reading file: {e}")
            else:
                print(f"File not found: {file_path}")
        else:
            result = clean_file(
                file_path,
                output_path=args.output if len(args.files) == 1 else None,
                backup=not args.no_backup
            )
            if result >= 0:
                total_replacements += result

    if not args.dry_run:
        print(f"\n{'='*40}")
        print(f"Total replacements across all files: {total_replacements}")

if __name__ == '__main__':
    main()