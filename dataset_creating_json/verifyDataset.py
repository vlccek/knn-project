import os
import json
import argparse
import sys

# Názvy adresářů pro jednotlivé splity (očekávané)
EXPECTED_SPLITS = ["train", "test", "validation"]

def verify_split(dataset_base_dir, split_name):
    """
    Ověří integritu jednoho splitu datasetu.

    Zkontroluje, zda každý .jpg soubor v adresáři splitu má odpovídající
    záznam v metadata.jsonl souboru tohoto splitu.

    Args:
        dataset_base_dir (str): Cesta k hlavnímu adresáři datasetu (např. './dataset').
        split_name (str): Název splitu k ověření (např. 'train').

    Returns:
        bool: True, pokud je split v pořádku, jinak False.
    """
    print(f"--- Verifying split: {split_name} ---")
    split_dir = os.path.join(dataset_base_dir, split_name)
    metadata_path = os.path.join(split_dir, "metadata.jsonl")
    split_ok = True

    # 1. Zkontrolovat existenci adresáře a metadata souboru
    if not os.path.isdir(split_dir):
        print(f"  ERROR: Split directory not found: {split_dir}")
        return False # Tento split nelze ověřit

    if not os.path.isfile(metadata_path):
        print(f"  ERROR: Metadata file not found: {metadata_path}")
        # Pokud metadata neexistují, ale adresář ano, můžeme zkontrolovat, zda jsou v něm obrázky
        try:
            dir_images_check = [f for f in os.listdir(split_dir)
                                if os.path.isfile(os.path.join(split_dir, f)) and f.lower().endswith('.jpg')]
            if dir_images_check:
                print(f"  ERROR: Found {len(dir_images_check)} image(s) in {split_dir}, but no metadata file.")
                return False
            else:
                print(f"  INFO: Split directory {split_dir} is empty and metadata file is missing. Assuming ok.")
                return True # Prázdný adresář bez metadat je technicky "validní" v kontextu této kontroly
        except Exception as e:
             print(f"  ERROR: Could not check directory contents {split_dir}: {e}")
             return False


    # 2. Získat seznam obrázků (.jpg) fyzicky v adresáři splitu
    try:
        dir_images = set(f for f in os.listdir(split_dir)
                         if os.path.isfile(os.path.join(split_dir, f)) and f.lower().endswith('.jpg'))
        print(f"  Found {len(dir_images)} '.jpg' files in directory {split_dir}")
    except Exception as e:
        print(f"  ERROR: Could not list images in directory {split_dir}: {e}")
        return False # Nelze pokračovat bez seznamu obrázků

    # 3. Získat seznam obrázků z metadata.jsonl souboru
    metadata_images = set()
    try:
        with open(metadata_path, "r", encoding="utf-8") as f_meta:
            for line_num, line in enumerate(f_meta, 1):
                line = line.strip()
                if not line: # Přeskočit prázdné řádky
                    continue
                try:
                    record = json.loads(line)
                    if "file_name" in record:
                        metadata_images.add(record["file_name"])
                    else:
                        print(f"  WARNING: Missing 'file_name' key in {metadata_path}, line {line_num}. Line content: '{line[:100]}...'")
                        # Můžeme se rozhodnout, zda je to chyba -> split_ok = False
                except json.JSONDecodeError:
                    print(f"  ERROR: Invalid JSON found in {metadata_path}, line {line_num}. Line content: '{line[:100]}...'")
                    split_ok = False # Chyba v JSON je závažná
        print(f"  Found {len(metadata_images)} entries in metadata file {metadata_path}")

    except FileNotFoundError:
        # Tato chyba by již neměla nastat díky kontrole výše, ale pro jistotu
        print(f"  ERROR: Metadata file disappeared during processing: {metadata_path}")
        return False
    except Exception as e:
        print(f"  ERROR: Could not read or parse {metadata_path}: {e}")
        return False

    # 4. Porovnání - Najít obrázky v adresáři, které nejsou v metadatech
    images_in_dir_not_in_meta = dir_images - metadata_images
    if images_in_dir_not_in_meta:
        print(f"\n  ERROR: Found {len(images_in_dir_not_in_meta)} image file(s) in the '{split_name}' directory that are NOT listed in '{metadata_path}':")
        for filename in sorted(list(images_in_dir_not_in_meta))[:10]: # Vypíše max prvních 10
             print(f"    - {filename}")
        if len(images_in_dir_not_in_meta) > 10:
            print(f"    ... and {len(images_in_dir_not_in_meta) - 10} more.")
        split_ok = False
    else:
        print(f"  OK: All {len(dir_images)} image(s) in the '{split_name}' directory are listed in the metadata file.")

    # 5. (Volitelné) Porovnání - Najít záznamy v metadatech, pro které neexistuje obrázek
    images_in_meta_not_in_dir = metadata_images - dir_images
    if images_in_meta_not_in_dir:
        print(f"\n  WARNING: Found {len(images_in_meta_not_in_dir)} entries in '{metadata_path}' for which the corresponding image file is MISSING in the '{split_name}' directory:")
        for filename in sorted(list(images_in_meta_not_in_dir))[:10]: # Vypíše max prvních 10
            print(f"    - {filename}")
        if len(images_in_meta_not_in_dir) > 10:
            print(f"    ... and {len(images_in_meta_not_in_dir) - 10} more.")
        # Můžeme se rozhodnout, zda je toto chyba -> split_ok = False
        # Prozatím ponecháno jako WARNING

    if split_ok and not images_in_dir_not_in_meta:
         print(f"--- Verification for split '{split_name}' PASSED ---")
    else:
         print(f"--- Verification for split '{split_name}' FAILED ---")

    return split_ok


def main():
    parser = argparse.ArgumentParser(description="Verify dataset integrity: Check if all images in split folders exist in their respective metadata.jsonl.")
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the main dataset directory (containing train, test, validation subdirectories).",
        default="./dataset", # Výchozí hodnota
        nargs='?' # Argument je volitelný
    )
    args = parser.parse_args()

    dataset_path = args.dataset_dir

    print(f"Starting verification for dataset directory: {dataset_path}")

    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset directory not found: {dataset_path}")
        sys.exit(1)

    overall_status = True
    for split in EXPECTED_SPLITS:
        split_status = verify_split(dataset_path, split)
        overall_status = overall_status and split_status # Pokud je jakýkoliv split False, celkový status bude False
        print("-" * 30) # Oddělovač mezi splity

    print("\n===== Verification Summary =====")
    if overall_status:
        print("✅ All checks passed. Dataset integrity verified successfully!")
        sys.exit(0) # Ukončení s kódem 0 (úspěch)
    else:
        print("❌ Verification failed. Found discrepancies in the dataset.")
        sys.exit(1) # Ukončení s kódem 1 (chyba)


if __name__ == "__main__":
    main()