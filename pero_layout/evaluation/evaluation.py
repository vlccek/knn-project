import os
import json
import zss
from zss import Node
from typing import Any, Dict, List, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path_json_predction = "../pipeline/output/structure"
path_json_gt = "../../dataset_creating_json/responses/"

import os
import json
import numpy as np


def edit_distance(s1, s2):
    """
    Vypočítá Levenshteinovu vzdálenost mezi dvěma řetězci.

    Args:
        s1: První řetězec
        s2: Druhý řetězec

    Returns:
        int: Počet operací potřebných k transformaci s1 na s2
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def json_to_tree(json_obj, parent=None, is_leaf=False):
    """
    Převede JSON objekt na stromovou strukturu pro knihovnu ZSS.

    Args:
        json_obj: JSON objekt k převodu
        parent: Rodičovský uzel (None pro kořen)
        is_leaf: Určuje, zda je uzel listem stromu

    Returns:
        zss.Node: Stromová struktura reprezentující JSON
    """
    if parent is None:
        # Vytvoření kořenového uzlu
        if isinstance(json_obj, dict):
            parent = zss.Node("dict")
        elif isinstance(json_obj, list):
            parent = zss.Node("list")
        else:
            # Listy jsou označeny speciálním tagem
            return zss.Node(f"<leaf>{str(json_obj)}")

    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            key_node = zss.Node(key)
            parent.addkid(key_node)

            if isinstance(value, (dict, list)):
                json_to_tree(value, key_node)
            else:
                # Hodnota je list
                value_node = zss.Node(f"<leaf>{str(value)}")
                key_node.addkid(value_node)

    elif isinstance(json_obj, list):
        for i, item in enumerate(json_obj):
            idx_node = zss.Node(f"[{i}]")
            parent.addkid(idx_node)

            if isinstance(item, (dict, list)):
                json_to_tree(item, idx_node)
            else:
                # Item je list
                item_node = zss.Node(f"<leaf>{str(item)}")
                idx_node.addkid(item_node)

    return parent


def calculate_ted(tree1, tree2):
    """
    Vypočítá normalizovanou Tree Edit Distance (TED) mezi dvěma JSON strukturami
    s využitím knihovny ZSS.

    Args:
        tree1: První JSON objekt
        tree2: Druhý JSON objekt

    Returns:
        float: Normalizovaná hodnota TED mezi stromy (0-1)
    """
    # Převedení JSON na stromové struktury pro ZSS
    zss_tree1 = json_to_tree(tree1)
    zss_tree2 = json_to_tree(tree2)

    # Definice vah pro operace podle poskytnutého vzoru
    def insert_cost(node):
        """
        Insert cost pro uzly. Pro listy je to délka hodnoty, jinak 1.
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def remove_cost(node):
        """
        Remove cost pro uzly. Pro listy je to délka hodnoty, jinak 1.
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def update_cost(node1, node2):
        """
        Update cost pro uzly podle vzoru.
        - Pokud jsou oba listy, počítá se editační vzdálenost mezi hodnotami
        - Pokud je jeden list a druhý ne, je to délka hodnoty listu + 1
        - Jinak je to 0 pokud jsou stejné, 1 pokud jsou různé
        """
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2

        if label1_leaf == True and label2_leaf == True:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    # Výpočet absolutní TED pomocí knihovny ZSS
    ted = zss.distance(zss_tree1, zss_tree2,
                       get_children=zss.Node.get_children,
                       insert_cost=insert_cost,
                       remove_cost=remove_cost,
                       update_cost=update_cost)

    # Normalizace výsledku do rozsahu 0-1
    # Vypočítáme teoretickou maximální vzdálenost
    # Pro lepší normalizaci počítáme "velikost stromu" s ohledem na definované váhy
    def weighted_size(node):
        if "<leaf>" in node.label:
            leaf_value = len(node.label.replace("<leaf>", ""))
            return leaf_value if leaf_value > 0 else 1
        else:
            return 1 + sum(weighted_size(child) for child in node.children)

    size1 = weighted_size(zss_tree1)
    size2 = weighted_size(zss_tree2)

    # Teoreticky nejhorší případ: smaž jeden strom a vlož druhý
    max_possible_ted = size1 + size2

    # Normalizace na rozsah 0-1
    if max_possible_ted == 0:  # Oba stromy jsou prázdné
        return 0
    else:
        return ted / max_possible_ted


def normalize_dict(d):
    """ Normalizuje hodnoty v dictu – trim, lowercase atd. podle potřeby."""
    if isinstance(d, dict):
        return {k: normalize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [normalize_dict(i) for i in d]
    elif isinstance(d, str):
        return d.strip().lower()  # příklad normalizace
    else:
        return d

def flatten(d, parent_key='', sep='.'):
    """Zploští vnořený dict do flat tvaru – např. {'a.b': value}"""
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten(v, new_key, sep=sep))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}[{i}]"
            items.extend(flatten(v, new_key, sep=sep))
    else:
        items.append(f"{parent_key}={str(d)}")
    return items


def calc_f1(preds: List[dict], answers: List[dict]):
    """Field-level micro-averaged F1 podle tebou zadané definice"""
    total_tp, total_fn_or_fp = 0, 0
    for pred, answer in zip(preds, answers):
        pred_flat = set(flatten(normalize_dict(pred)))
        answer_flat = set(flatten(normalize_dict(answer)))
        for field in pred_flat:
            if field in answer_flat:
                total_tp += 1
                answer_flat.remove(field)
            else:
                total_fn_or_fp += 1
        total_fn_or_fp += len(answer_flat)  # zbylé nenalezené v predikci
    if total_tp + 0.5 * total_fn_or_fp == 0:
        return 0.0
    return total_tp / (total_tp + 0.5 * total_fn_or_fp)




if __name__ == "__main__":
    # Get the list of JSON files in the specified directory
    json_files = [f for f in os.listdir(path_json_predction) if f.endswith('.json')]

    obj = {
        "predicted": [],
        "ground_truth": [],
        "ted_value": [],
        "file_name": []
    }
    # Iterate through each JSON file
    for json_file in json_files:
        # Construct the full file path
        file_path = os.path.join(path_json_predction, json_file)
        gt_file_path = os.path.join(path_json_gt, json_file)

        # Open and load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)


        with open(gt_file_path, 'r') as file:
            gt_data = json.load(file)



        # Výpočet TED mezi predikcí a ground truth
        ted_value = calculate_ted(data, gt_data)

        ted = 1 - ted_value


        obj["predicted"].append(data)
        obj["ground_truth"].append(gt_data)
        obj["ted_value"].append(ted)
        obj["file_name"].append(json_file)

        # Výpis hodnoty TED pro tento soubor
        print(f"Tree Edit Distance pro soubor {json_file}: {1 - ted_value:.2f}")

    f1_all = calc_f1(obj["predicted"], obj["ground_truth"])
    print(f"\nGlobální F1 (field-level, micro-averaged): {f1_all:.4f}")


    # Výpis statistik TED
    df= pd.DataFrame(obj)
    df.to_csv("ted_results.csv", index=False)
    print(df["ted_value"].describe())

    # plot histogram

    plt.figure(figsize=(10, 6))
    sns.histplot(df["ted_value"], bins=10, kde=True)
    plt.title('Histogram TED Values')
    plt.savefig("ted_histogram.pdf")

