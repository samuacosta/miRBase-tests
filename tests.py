#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''miRBase tests

miRBase tests and utilities.
'''
import numpy
import sys
import random
import subprocess
import re
import decimal
import math
import os
import shutil
import time
import types
import uuid
import argparse
import datetime
import subprocess
import operator
import collections
import pandas
from datetime import datetime
from sklearn import metrics
from shutil import copy

__author__ = "Samuel Acosta"
__email__ = "samuel.acostamelgarejo@postgrad.manchester.ac.uk"
__version__ = "1.0.0"


class Mirna(object):
    def __init__(self, id=None, ac=None, cluster=None, cluster_nr=None):
        self.id = id
        self.ac = ac
        self.cluster = cluster
        self.cluster_nr = cluster_nr

    def __eq__(self, other):
        return self.id == other.id


def main():
    '''Main method'''
    try:
        start = datetime.now()
        print("------ Running")
        print()

        args = read_parameters()

        dataset_filename = args.file  # input/hairpin.fa
        db_filename = bash_format_db_for_blast(dataset_filename)
        blast_filename = bash_blastall_db(db_filename)
        mcl_filename = bash_run_mcl(blast_filename)
        # Another one with custom BLAST
        '''
        word_size = "7"  # def 11
        reward = "5"  # def 1
        penalty = "-4"  # def -3
        blast_filename_custom = bash_blastall_db_custom(
            db_filename, word_size, reward, penalty)
        mcl_filename_custom = bash_run_mcl(blast_filename_custom)
        print()
        '''

        '''
        mcl_filename = "output/m8.cblast.mcl"
        mcl_filename_custom = "output/m8.cblast.custom.mcl"
        '''

        aliases = load_aliases("input/aliases.txt")
        # Gets mirnas with their clusters from mirbase and adds unclustered
        mirbase_families = get_families_with_unclustered_mirbase(
            "input/miFam.dat", "input/hairpin.fa")
        # Updates the list with the numeric mappings and returns a mapping dictionary
        mirbase_families, numeric_cluster_mappings = get_numeric_cluster_mappings(
            mirbase_families)
        mcl_families = get_families_from_mcl(
            mcl_filename, aliases, numeric_cluster_mappings)
        # mcl_families_custom = get_families_from_mcl(
        #   mcl_filename_custom, aliases, numeric_cluster_mappings)

        numeric_cluster_ids_mirbase = [
            mirna.cluster_nr for mirna in mirbase_families]
        numeric_cluster_ids_mcl = [
            mirna.cluster_nr for mirna in mcl_families]

        print("--- MCL with default BLAST")
        run_metrics(numeric_cluster_ids_mirbase,
                    numeric_cluster_ids_mcl, mcl_families)
        print()
        '''
        print("--- MCL with custom BLAST")
        run_metrics(numeric_cluster_ids_mirbase,
                    numeric_cluster_ids_mcl, mcl_families_custom)
        '''

        '''
        write_families(mirbase_families, "output/miFam.dat.sorted")
        write_families(mcl_families, "output/miFamMcl.dat.sorted")
        write_families_labels_only(
            mirbase_families, "output/miFam.dat.sorted.labels")
        write_families_labels_only(
            mcl_families, "output/miFamMcl.dat.sorted.labels")
        '''

        print("------ Finished succesfully (", datetime.now()-start, ")")
    except Exception as e:
        print("An error occurred during execution:")
        print(e)
        raise


def read_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--file",
                        nargs='*',
                        help="The name of the dataset to cluster. [Default=hairpin.fa]",
                        default="input/hairpin.fa"
                        )
    return(parser.parse_args())


def bash_format_db_for_blast(dataset_filename):
    db_filename = dataset_filename.replace("input/", "output/") + ".db"
    copy(dataset_filename, db_filename)
    template = "formatdb -i #db_filename# -p F"
    command = template.replace("#db_filename#", db_filename)
    print("--- Running formatdb, output:")
    print(subprocess.getoutput(command))
    return db_filename


def bash_blastall_db(db_filename):
    blast_filename = "output/m8.cblast"
    template = "blastall -i #db_filename# -d #db_filename# -p blastn -m 8 > #blast_filename#"
    command = template.replace("#db_filename#", db_filename).replace(
        "#blast_filename#", blast_filename)
    print()
    print("--- Running blastall")
    print(subprocess.getoutput(command))
    return blast_filename


def bash_blastall_db_custom(db_filename, word_size, reward, penalty):
    blast_filename = "output/m8.cblast.custom"
    template = "blastall -i #db_filename# -d #db_filename# -p blastn -X #word_size# -r #reward# -q #penalty# -m 8 > #blast_filename#"
    command = template.replace("#db_filename#", db_filename).replace(
        "#blast_filename#", blast_filename).replace("#word_size#", word_size).replace(
            "reward", reward).replace("#penalty#", penalty)
    print()
    print("--- Running blastall")
    print(subprocess.getoutput(command))
    return blast_filename


def bash_run_mcl(blast_filename):
    mcl_filename = blast_filename + ".mcl"
    template = "mcxdeblast --m9 --line-mode=abc --out=- #blast_filename# | mcl - --abc -o #mcl_filename#"
    command = template.replace("#blast_filename#", blast_filename).replace(
        "#mcl_filename#", mcl_filename)
    print("--- Running MCL")
    print(subprocess.getoutput(command))
    return mcl_filename


def load_aliases(aliases_filename):
    aliases = {}
    with open(aliases_filename) as file:
        for line in file:
            (alias, mirnas) = line.split("\t")
            for mirna in mirnas.split(";"):
                if mirna != "" and mirna != "\n":
                    aliases[mirna] = alias
    return aliases


def get_families_from_mcl(mcl_filename, aliases, numeric_cluster_mappings):
    mcl_families = []
    cluster_ids = {}
    with open(mcl_filename) as inputfile:
        for line in inputfile:
            mirnas_str = line.split("\t")
            # Unclustered mirnas
            if len(mirnas_str) == 1:
                cluster_id = "uc_" + mirnas_str[0].strip("\n")

            else:
                mirnas_subids = [
                    mirna_str[mirna_str.index("-")+1:].strip("\n") for mirna_str in mirnas_str]
                cluster_id = get_cluster_id(mirnas_subids)

                # Handle duplicate family ids
                if cluster_id in cluster_ids.keys():
                    nr_duplicate = cluster_ids[cluster_id] + 1
                    if "_" not in cluster_id:
                        # First duplicate, rename previous cluster_id with "_1"
                        for mirna in mcl_families:
                            if mirna.cluster == cluster_id:
                                mirna.cluster = cluster_id + "_1"
                        cluster_ids[cluster_id +
                                    "_1"] = cluster_ids.pop(cluster_id)
                        cluster_ids[cluster_id] = nr_duplicate
                        nr_duplicate = nr_duplicate + 1
                    cluster_id = cluster_id.split(
                        "_")[0] + "_" + str(nr_duplicate)

                cluster_ids[cluster_id] = 0

            cluster_nr = numeric_cluster_mappings.get(cluster_id)
            if cluster_nr is None:
                cluster_nr = max(numeric_cluster_mappings.values()) + 1
                numeric_cluster_mappings[cluster_id] = cluster_nr

            for mirna_str in mirnas_str:
                mirna = Mirna()
                mirna.id = mirna_str.strip("\n")
                mirna.ac = aliases[mirna_str.strip("\n")]
                mirna.cluster = cluster_id
                mirna.cluster_nr = cluster_nr
                mcl_families.append(mirna)

    return mcl_families


def get_cluster_id(mirnas_subids):
    for idx, subid in enumerate(mirnas_subids):
        if str.isalpha(subid[-1]) and str.isnumeric(subid[-2]):
            mirnas_subids[idx] = subid[:-1]
    prefix_ranking = sorted([(sum([w.startswith(prefix)
                                   for w in mirnas_subids]), prefix) for prefix in mirnas_subids])[::-1]
    return prefix_ranking[0][1].strip("\n")


def get_families_with_unclustered_mirbase(clustered_filename, all_filename):
    mirbase_families, mirbase_families_ids = get_families_from_mirbase(
        clustered_filename)
    all_mirnas_list = add_unclustered_mirnas(
        mirbase_families, mirbase_families_ids, all_filename)
    return all_mirnas_list


def get_families_from_mirbase(clustered_filename):
    mirna_families = []
    mirna_families_ids = []
    with open(clustered_filename) as inputfile:
        cluster_id = ""
        for line in inputfile:
            if line.startswith("AC") or line.startswith("//") or line == "":  # ignore
                continue
            elif line.startswith("ID"):
                cluster_id = line[5:].strip("\n")
            else:
                mirna = Mirna()
                mirna.ac = line.split("  ")[1]
                mirna.id = line.split("  ")[2].strip("\n")
                mirna.cluster = cluster_id
                mirna_families.append(mirna)
                mirna_families_ids.append(line.split("  ")[2].strip("\n"))
    return mirna_families, mirna_families_ids


def add_unclustered_mirnas(mirbase_families, mirbase_families_ids, all_filename):
    with open(all_filename) as inputfile:
        for line in inputfile.readlines():
            if line.startswith(">"):
                mirna_id = line.split(" ")[0][1:]
                if mirna_id not in mirbase_families_ids:
                    mirna = Mirna()
                    mirna.id = mirna_id
                    mirna.ac = line.split(" ")[1]
                    mirna.cluster = "uc_" + line.split(" ")[0][1:]
                    mirbase_families.append(mirna)
    return mirbase_families


def get_numeric_cluster_mappings(mirbase_families):
    cluster_nr = 0
    last_cluster = ""
    numeric_cluster_mappings = {}
    for mirna in mirbase_families:
        if last_cluster != mirna.cluster:
            mirna.cluster_nr = cluster_nr
            cluster_nr += 1
            last_cluster = mirna.cluster
            numeric_cluster_mappings[mirna.cluster] = cluster_nr
        else:
            mirna.cluster_nr = cluster_nr
    return mirbase_families, numeric_cluster_mappings


def get_numeric_cluster_ids_mirbase(cluster_ids_mirbase_dict_sorted):
    numeric_cluster_ids_mirbase = []
    for key in cluster_ids_mirbase_dict_sorted:
        numeric_cluster_ids_mirbase.append(
            cluster_ids_mirbase_dict_sorted[key])
    return numeric_cluster_ids_mirbase


def get_numeric_cluster_ids_mcl(cluster_ids_mcl_dict_sorted, cluster_ids_mirbase_dict_sorted):
    numeric_cluster_ids_mcl = []
    for key in cluster_ids_mirbase_dict_sorted:
        if key in cluster_ids_mcl_dict_sorted:
            numeric_cluster_ids_mcl.append(cluster_ids_mcl_dict_sorted[key])
        else:
            # Dodgy TODO
            numeric_cluster_ids_mcl.append(99999999)
    return numeric_cluster_ids_mcl


def run_metrics(numeric_cluster_ids_mirbase, numeric_cluster_ids_mcl, mcl_families):

    print("--- Scores WITHOUT proper predicted-to-real cluster mapping")
    print_metrics(numeric_cluster_ids_mirbase, numeric_cluster_ids_mcl)

    cluster_names_mcl = [mirna.cluster for mirna in mcl_families]

    print()
    print("--- Scores WITH proper predicted-to-real cluster mapping")

    print("--- Normalization = ALL")
    equivalences_dict = get_cluster_equivalences(
        numeric_cluster_ids_mirbase, cluster_names_mcl, "all")
    numeric_cluster_ids_mcl = [
        equivalences_dict[mirna.cluster] for mirna in mcl_families]
    print_metrics(numeric_cluster_ids_mirbase, numeric_cluster_ids_mcl)

    print("--- Normalization = ROW")
    equivalences_dict = get_cluster_equivalences(
        numeric_cluster_ids_mirbase, cluster_names_mcl, "index")
    numeric_cluster_ids_mcl = [
        equivalences_dict[mirna.cluster] for mirna in mcl_families]
    print_metrics(numeric_cluster_ids_mirbase, numeric_cluster_ids_mcl)

    print("--- Normalization = COLUMNS")
    equivalences_dict = get_cluster_equivalences(
        numeric_cluster_ids_mirbase, cluster_names_mcl, "columns")
    numeric_cluster_ids_mcl = [
        equivalences_dict[mirna.cluster] for mirna in mcl_families]
    print_metrics(numeric_cluster_ids_mirbase, numeric_cluster_ids_mcl)


def get_cluster_equivalences(numeric_cluster_ids_mirbase, cluster_names_mcl, normalization_mode):
    cluster_equivalences = {}
    df = pandas.DataFrame(
        {'Real': numeric_cluster_ids_mirbase, 'Pred': cluster_names_mcl})
    ct = pandas.crosstab(df['Real'], df['Pred'], normalize=normalization_mode)
    ct_sorted = ct.unstack().sort_values(ascending=False)
    ct_sorted = ct_sorted[ct_sorted > 0]
    print(ct_sorted.size)
    for indexes, count in ct_sorted.items():
        if indexes[0] not in cluster_equivalences:
            cluster_equivalences[indexes[0]] = indexes[1]

    value_new_clusters = max(numeric_cluster_ids_mirbase) + 1
    for cluster_mcl in cluster_names_mcl:
        if cluster_mcl not in cluster_equivalences:
            cluster_equivalences[cluster_mcl] = value_new_clusters
            value_new_clusters += 1

    return cluster_equivalences


def print_metrics(numeric_cluster_ids_mirbase, numeric_cluster_ids_mcl):
    print("ARI: ", metrics.adjusted_rand_score(
        numeric_cluster_ids_mirbase, numeric_cluster_ids_mcl))
    print("Fowlkes-Mallows: ", metrics.fowlkes_mallows_score(
        numeric_cluster_ids_mirbase, numeric_cluster_ids_mcl))


def write_families(families, output_filename):
    outputfile = open(output_filename, 'w')
    families_sorted = sorted(
        families, key=operator.attrgetter("cluster"))
    cluster = ""
    first_line = True
    for mirna in families_sorted:
        if cluster != mirna.cluster:
            cluster = mirna.cluster
            if not first_line:
                print("//", file=outputfile)
            first_line = False
            print(cluster, file=outputfile)
        print(mirna.id, file=outputfile)


def write_families_labels_only(families, output_filename):
    outputfile = open(output_filename, 'w')
    families_sorted = sorted(
        families, key=operator.attrgetter("cluster"))
    cluster = ""
    for mirna in families_sorted:
        if cluster != mirna.cluster:
            cluster = mirna.cluster
            print(cluster, file=outputfile)


# Entry
if __name__ == "__main__":
    main()
