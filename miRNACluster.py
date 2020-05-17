#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''miRNACluster

Unsupervised microRNA clustering comparison tool.
'''
import argparse
import datetime
import decimal
import operator
import shutil
import subprocess

import pandas
import sklearn
import sklearn.cluster

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
        start = datetime.datetime.now()
        print("------ Running")
        print()

        args = read_parameters()
        dataset_filename = args.file
        if args.algorithms is None:
            algorithms = []
        else:
            algorithms = args.algorithms
        if args.cached is None:
            steps_with_cached_files = []
        else:
            steps_with_cached_files = args.cached
        include_raw_files = args.raw

        db_filename = dataset_filename.replace("input/", "output/") + ".db"
        if "formatdb" not in steps_with_cached_files:
            bash_format_db_for_blast(dataset_filename, db_filename)

        blast_filename = "output/m8.cblast"
        if "blast" not in steps_with_cached_files:
            word_size = "7"  # def 11
            reward = "5"  # def 1
            penalty = "-4"  # def -3
            cost_gap_open = "10"  # def 5
            cost_gap_extend = "6"  # def 2
            # bash_blastall_db(db_filename)
            bash_blastall_db_custom(
                db_filename, blast_filename, word_size, reward, penalty, cost_gap_open, cost_gap_extend)

        aliases_filename = "input/aliases.txt"
        mirna_clustered_filename = "input/miFam.dat"

        # Load aliases
        aliases = load_aliases(aliases_filename)
        # Load clustered families from mirbase
        mirbase_families_clustered, mirbase_families_ids = get_families_from_mirbase(
            mirna_clustered_filename)
        # Add unclustered mirnas from mirbase
        mirbase_families = add_unclustered_mirnas(
            mirbase_families_clustered, mirbase_families_ids, dataset_filename)
        # Sort
        mirbase_families_sorted = sorted(
            mirbase_families, key=operator.attrgetter('id'))
        # Assign numbers to clusters
        mirbase_families = get_numeric_cluster_mappings(mirbase_families)
        # Get the cluster numbers
        numeric_cluster_ids_mirbase = [
            mirna.cluster_nr for mirna in mirbase_families_sorted]

        print("\n----------------------")
        print("--- miRBase families")
        print_stats(mirbase_families)
        mirbase_raw_file = mirna_clustered_filename.replace(
            "input", "output").replace(".dat", "")
        file_outputs_mirbase_families(mirbase_families, mirbase_raw_file)
        print()

        # blast_filename = "output/m8.cblast.custom.test"
        if "density" in algorithms:
            # Get distance and affinity matrices from blast results
            mirna_ids, distance_matrix = get_matrices_from_blast(
                blast_filename)

        if "centroid" in algorithms:
            print("--- Running centroid algorithm")
            cached = False
            if "clustal" in steps_with_cached_files:
                cached = True
            clustal_filename = "output/clustal.out"
            clustal_families = get_families_from_clustal(
                dataset_filename, clustal_filename, aliases, mirbase_families, cached)
            clustal_families_sorted = sorted(
                clustal_families, key=operator.attrgetter('id'))

            print("\n----------------------")
            print("--- CLUSTAL families")
            print_stats(clustal_families)
            cluster_equivalences = run_metrics(
                numeric_cluster_ids_mirbase, clustal_families_sorted)
            families_filename = "output/miFamCentroid"
            file_outputs(clustal_families_sorted, cluster_equivalences,
                         mirbase_families, include_raw_files, families_filename)

        if "graph" in algorithms:
            mcl_inflation_value = "1.2"
            mcl_filename = blast_filename + \
                mcl_inflation_value.replace(".", "") + ".mcl"
            if "mcl" not in steps_with_cached_files:
                # bash_run_mcl(blast_filename, mcl_filename)
                bash_run_mcl_custom(
                    blast_filename, mcl_filename, mcl_inflation_value)
            mcl_families = get_families_from_mcl(
                mcl_filename, aliases, mirbase_families)
            mcl_families_sorted = sorted(
                mcl_families, key=operator.attrgetter('id'))
            # numeric_cluster_ids_mcl = [
            #    mirna.cluster_nr for mirna in mcl_families_sorted]

            print("\n----------------------")
            print("--- MCL families")
            print_stats(mcl_families)
            cluster_equivalences = run_metrics(
                numeric_cluster_ids_mirbase, mcl_families_sorted)
            families_filename = "output/miFamGraph"
            file_outputs(mcl_families_sorted, cluster_equivalences,
                         mirbase_families, include_raw_files, families_filename)

        if "density" in algorithms:
            print("--- Running density algorithm")
            eps = 0.000071
            dbscan = sklearn.cluster.DBSCAN(
                eps=eps, metric="precomputed", min_samples=2).fit(distance_matrix)
            print()

            dbscan_families = get_families_from_dbscan(
                mirna_ids, dbscan.labels_, aliases, mirbase_families)
            dbscan_families_sorted = sorted(
                dbscan_families, key=operator.attrgetter('id'))

            print("\n----------------------")
            print("--- DBSCAN families")
            print_stats(dbscan_families)
            cluster_equivalences = run_metrics(
                numeric_cluster_ids_mirbase, dbscan_families_sorted)
            families_filename = "output/miFamDensity"
            file_outputs(dbscan_families_sorted, cluster_equivalences,
                         mirbase_families, include_raw_files, families_filename)

        print()
        print("------ Finished succesfully (", datetime.datetime.now()-start, ")")
    except Exception as exception:
        print("An error occurred during execution:")
        print(exception)
        raise


def read_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--file",
                        nargs='*',
                        help="The name of the dataset to cluster. [Default=hairpin.fa]",
                        default="input/hairpin.fa"
                        )
    parser.add_argument("-a",
                        "--algorithms",
                        nargs='*',
                        help="List of algorithms to run (centroid/graph/density). [Default=all]",
                        default=["centroid", "graph", "density"]
                        )
    parser.add_argument("-c",
                        "--cached",
                        nargs='*',
                        help="List of steps where a cached file is assumed (formatdb/blast/clustal/mcl). [Default=none]",
                        #default=["formatdb", "blast", "clustal", "mcl"]
                        )
    parser.add_argument("-r",
                        "--raw",
                        action='store_false',
                        help="Include the raw files for miRNA families, useful for comparison. [Default=True]",
                        )
    return(parser.parse_args())


def bash_format_db_for_blast(dataset_filename, db_filename):
    shutil.copy(dataset_filename, db_filename)
    template = "formatdb -i #db_filename# -p F"
    command = template.replace("#db_filename#", db_filename)
    print("--- Running formatdb, output:")
    print(subprocess.getoutput(command))
    return db_filename


def bash_blastall_db(db_filename, blast_filename):
    template = "blastall -i #db_filename# -d #db_filename# -p blastn -m 8 > #blast_filename#"
    command = template.replace("#db_filename#", db_filename).replace(
        "#blast_filename#", blast_filename)
    print()
    print("--- Running blastall")
    print(subprocess.getoutput(command))
    return blast_filename


def bash_blastall_db_custom(db_filename, blast_filename, word_size, reward, penalty, cost_gap_open, cost_gap_extend):
    template = "blastall -i #db_filename# -d #db_filename# -p blastn -X #word_size# -r #reward# -q #penalty# -G #cost_gap_open# -E #cost_gap_extend# -m 8 > #blast_filename#"
    command = template.replace("#db_filename#", db_filename)
    command = command.replace("#blast_filename#", blast_filename)
    command = command.replace("#word_size#", word_size)
    command = command.replace("#reward#", reward)
    command = command.replace("#penalty#", penalty)
    command = command.replace("#cost_gap_open#", cost_gap_open)
    command = command.replace("#cost_gap_extend#", cost_gap_extend)
    print()
    print("--- Running blastall")
    print(subprocess.getoutput(command))
    return blast_filename


def bash_run_mcl(blast_filename, mcl_filename):
    template = "mcxdeblast --m9 --line-mode=abc --out=- #blast_filename# | mcl - --abc -o #mcl_filename#"
    command = template.replace("#blast_filename#", blast_filename).replace(
        "#mcl_filename#", mcl_filename)
    print("--- Running MCL")
    print(subprocess.getoutput(command))
    print()
    return mcl_filename


def bash_run_mcl_custom(blast_filename, mcl_filename, inflation):
    template = "mcxdeblast --m9 --line-mode=abc --out=- #blast_filename# | mcl - --abc -I #inflation# -o #mcl_filename#"
    command = template.replace("#blast_filename#", blast_filename)
    command = command.replace("#mcl_filename#", mcl_filename)
    command = command.replace("#inflation#", inflation)
    print("--- Running graph algorithm")
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


def get_families_from_clustal(dataset_filename, clustal_filename, aliases, mirbase_families, cached):

    clustal_families = []
    cluster_ids = {}
    if not cached:
        run_clustal(dataset_filename, clustal_filename)

    with open(clustal_filename) as inputfile:
        cluster_id = ""
        mirnas_str = []
        first_line = True
        cluster_lbl = ""
        for line in inputfile:
            if line != "" and line != "\n":
                cluster_lbl_new = line.split(" ")[1].replace(":", "")
                if cluster_lbl_new != cluster_lbl and not first_line:
                    mirnas_subids = [mirna_str[mirna_str.index(
                        "-")+1:] for mirna_str in mirnas_str]

                    cluster_id = ""
                    if len(mirnas_str) == 1:
                        # Unclustered mirnas
                        cluster_id = "uc_" + mirnas_str[0]
                    else:
                        cluster_id = get_cluster_id(mirnas_subids)
                    # Handle duplicate family ids
                        if cluster_id in cluster_ids.keys():
                            nr_duplicate = cluster_ids[cluster_id] + 1
                            if "_" not in cluster_id:
                                # First duplicate, rename previous cluster_id with "_1"
                                for mirna in clustal_families:
                                    if mirna.cluster == cluster_id:
                                        mirna.cluster = cluster_id + "_1"
                                cluster_ids[cluster_id +
                                            "_1"] = cluster_ids.pop(cluster_id)
                                cluster_ids[cluster_id] = nr_duplicate
                                nr_duplicate = nr_duplicate + 1
                            cluster_id = cluster_id.split(
                                "_")[0] + "_" + str(nr_duplicate)

                    cluster_ids[cluster_id] = 0

                    for mirna_str in mirnas_str:
                        mirna = Mirna()
                        mirna.id = mirna_str
                        mirna.ac = aliases[mirna_str]
                        mirna.cluster = cluster_id
                        clustal_families.append(mirna)
                    cluster_lbl = cluster_lbl_new
                    mirnas_str = []

                else:
                    if first_line:
                        first_line = False
                        cluster_lbl = cluster_lbl_new

                mirna_str = line.split(" ")[8]
                mirnas_str.append(mirna_str)

        if mirnas_str:
            # List is not empty, process the items of the last cluster
            mirnas_subids = [mirna_str[mirna_str.index(
                "-")+1:].strip("\n") for mirna_str in mirnas_str]

            cluster_id = ""
            if len(mirnas_str) == 1:
                # Unclustered mirna
                cluster_id = "uc_" + mirnas_str[0]
            else:
                cluster_id = get_cluster_id(mirnas_subids)
                # Handle duplicate family ids
                if cluster_id in cluster_ids.keys():
                    nr_duplicate = cluster_ids[cluster_id] + 1
                    if "_" not in cluster_id:
                        # First duplicate, rename previous cluster_id with "_1"
                        for mirna in clustal_families:
                            if mirna.cluster == cluster_id:
                                mirna.cluster = cluster_id + "_1"
                        cluster_ids[cluster_id +
                                    "_1"] = cluster_ids.pop(cluster_id)
                        cluster_ids[cluster_id] = nr_duplicate
                        nr_duplicate = nr_duplicate + 1
                    cluster_id = cluster_id.split(
                        "_")[0] + "_" + str(nr_duplicate)

            cluster_ids[cluster_id] = 0

            for mirna_str in mirnas_str:
                mirna = Mirna()
                mirna.id = mirna_str
                mirna.ac = aliases[mirna_str]
                mirna.cluster = cluster_id
                clustal_families.append(mirna)

    # Added for the cases where the BLASTing leaves out some miRNAs
    mirna_ids_mcl = [mirna.id for mirna in clustal_families]
    for mirna_mirbase in mirbase_families:
        if mirna_mirbase.id not in mirna_ids_mcl:
            mirna = Mirna()
            mirna.id = mirna_mirbase.id
            mirna.ac = mirna_mirbase.ac
            cluster_id = "uc_" + mirna_mirbase.id
            mirna.cluster = cluster_id
            clustal_families.append(mirna)

    return clustal_families


def get_families_from_mcl(mcl_filename, aliases, mirbase_families):
    mcl_families = []
    cluster_ids = {}
    with open(mcl_filename) as inputfile:
        for line in inputfile:
            mirnas_str = line.split("\t")

            cluster_id = ""
            if len(mirnas_str) == 1:
                # Unclustered mirnas
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

            # Old way, naive cluster_nr mapping using cluster name
            # Now mapped later with cluster similarity matrix
            # cluster_nr = numeric_cluster_mappings.get(cluster_id)
            # if cluster_nr is None:
            #    cluster_nr = max(numeric_cluster_mappings.values()) + 1
            #    numeric_cluster_mappings[cluster_id] = cluster_nr

            for mirna_str in mirnas_str:
                mirna = Mirna()
                mirna.id = mirna_str.strip("\n")
                mirna.ac = aliases[mirna_str.strip("\n")]
                mirna.cluster = cluster_id
                # mirna.cluster_nr = cluster_nr
                mcl_families.append(mirna)

    # Added for the cases where the BLASTing leaves out some miRNAs
    mirna_ids_mcl = [mirna.id for mirna in mcl_families]
    for mirna_mirbase in mirbase_families:
        if mirna_mirbase.id not in mirna_ids_mcl:
            mirna = Mirna()
            mirna.id = mirna_mirbase.id
            mirna.ac = mirna_mirbase.ac
            cluster_id = "uc_" + mirna_mirbase.id
            mirna.cluster = cluster_id
            # cluster_nr = max(numeric_cluster_mappings.values()) + 1
            # numeric_cluster_mappings[cluster_id] = cluster_nr
            # mirna.cluster_nr = cluster_nr
            mcl_families.append(mirna)

    return mcl_families


def get_families_from_dbscan(mirna_ids, algorithm_labels, aliases, mirbase_families):

    families = []
    cluster_ids = {}
    eq_dict = dict(zip(mirna_ids, algorithm_labels))

    mirnas_str = []
    cluster_id = ""
    cluster_lbl = ""
    first_line = True
    for mirna_id in sorted(eq_dict, key=eq_dict.get):
        algorithm_label = eq_dict[mirna_id]
        if (algorithm_label != cluster_lbl or algorithm_label == -1) and not first_line:
            mirnas_subids = [mirna_str[mirna_str.index(
                "-")+1:].strip("\n") for mirna_str in mirnas_str]

            cluster_id = ""
            if len(mirnas_str) == 1:
                # Unclustered mirnas
                cluster_id = "uc_" + mirnas_str[0]
            else:
                cluster_id = get_cluster_id(mirnas_subids)
                # Handle duplicate family ids
                if cluster_id in cluster_ids.keys():
                    nr_duplicate = cluster_ids[cluster_id] + 1
                    if "_" not in cluster_id:
                        # First duplicate, rename previous cluster_id with "_1"
                        for mirna in families:
                            if mirna.cluster == cluster_id:
                                mirna.cluster = cluster_id + "_1"
                        cluster_ids[cluster_id +
                                    "_1"] = cluster_ids.pop(cluster_id)
                        cluster_ids[cluster_id] = nr_duplicate
                        nr_duplicate = nr_duplicate + 1
                    cluster_id = cluster_id.split(
                        "_")[0] + "_" + str(nr_duplicate)

            cluster_ids[cluster_id] = 0

            for mirna_str in mirnas_str:
                mirna = Mirna()
                mirna.id = mirna_str.strip("\n")
                mirna.ac = aliases[mirna_str.strip("\n")]
                mirna.cluster = cluster_id
                families.append(mirna)

            mirnas_str = []
            cluster_lbl = algorithm_label
        else:
            first_line = False
        mirnas_str.append(mirna_id)

    if mirnas_str:
        # List is not empty, process the items of the last cluster
        mirnas_subids = [mirna_str[mirna_str.index(
            "-")+1:].strip("\n") for mirna_str in mirnas_str]

        cluster_id = ""
        if len(mirnas_str) == 1:
            # Unclustered mirna
            cluster_id = "uc_" + mirnas_str[0]
        else:
            cluster_id = get_cluster_id(mirnas_subids)
            # Handle duplicate family ids
            if cluster_id in cluster_ids.keys():
                nr_duplicate = cluster_ids[cluster_id] + 1
                if "_" not in cluster_id:
                    # First duplicate, rename previous cluster_id with "_1"
                    for mirna in families:
                        if mirna.cluster == cluster_id:
                            mirna.cluster = cluster_id + "_1"
                    cluster_ids[cluster_id +
                                "_1"] = cluster_ids.pop(cluster_id)
                    cluster_ids[cluster_id] = nr_duplicate
                    nr_duplicate = nr_duplicate + 1
                cluster_id = cluster_id.split(
                    "_")[0] + "_" + str(nr_duplicate)

        cluster_ids[cluster_id] = 0

        for mirna_str in mirnas_str:
            mirna = Mirna()
            mirna.id = mirna_str
            mirna.ac = aliases[mirna_str]
            mirna.cluster = cluster_id
            families.append(mirna)

    # Added for the cases where the BLASTing leaves out some miRNAs
    mirna_ids_algorithm = [mirna.id for mirna in families]
    for mirna_mirbase in mirbase_families:
        if mirna_mirbase.id not in mirna_ids_algorithm:
            mirna = Mirna()
            mirna.id = mirna_mirbase.id
            mirna.ac = mirna_mirbase.ac
            cluster_id = "uc_" + mirna_mirbase.id
            mirna.cluster = cluster_id
            families.append(mirna)

    return families


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
    # numeric_cluster_mappings = {}
    for mirna in mirbase_families:
        if last_cluster != mirna.cluster:
            mirna.cluster_nr = cluster_nr
            cluster_nr += 1
            last_cluster = mirna.cluster
            # numeric_cluster_mappings[mirna.cluster] = cluster_nr
        else:
            mirna.cluster_nr = cluster_nr
    return mirbase_families


def run_metrics(numeric_cluster_ids_mirbase, algorithm_families):

    cluster_names_algorithm = [mirna.cluster for mirna in algorithm_families]

    print("--- Normalization = ALL")
    equivalences_dict = get_cluster_equivalences(
        numeric_cluster_ids_mirbase, cluster_names_algorithm, "all")
    numeric_cluster_ids_algorithm = [
        equivalences_dict[mirna.cluster] for mirna in algorithm_families]
    print_metrics(numeric_cluster_ids_mirbase, numeric_cluster_ids_algorithm)
    print()

    '''
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
    '''
    return equivalences_dict


def get_cluster_equivalences(numeric_cluster_ids_mirbase, cluster_names_algorithm, normalization_mode):
    cluster_equivalences = {}
    dframe = pandas.DataFrame(
        {'Real': numeric_cluster_ids_mirbase, 'Pred': cluster_names_algorithm})
    dframe = pandas.crosstab(
        dframe['Real'], dframe['Pred'], normalize=normalization_mode)
    dframe = dframe.unstack().sort_values(ascending=False)
    dframe = dframe[dframe > 0]
    # print(ct_sorted.size)
    for indexes, count in dframe.items():
        if indexes[0] not in cluster_equivalences and indexes[1] not in cluster_equivalences.values():
            cluster_equivalences[indexes[0]] = indexes[1]

    value_new_clusters = max(numeric_cluster_ids_mirbase) + 1
    for cluster_algorithm in cluster_names_algorithm:
        if cluster_algorithm not in cluster_equivalences:
            cluster_equivalences[cluster_algorithm] = value_new_clusters
            value_new_clusters += 1

    return cluster_equivalences


def print_metrics(numeric_cluster_ids_mirbase, numeric_cluster_ids_mcl):
    print("ARI: ", sklearn.metrics.adjusted_rand_score(
        numeric_cluster_ids_mirbase, numeric_cluster_ids_mcl))
    print("Fowlkes-Mallows: ", sklearn.metrics.fowlkes_mallows_score(
        numeric_cluster_ids_mirbase, numeric_cluster_ids_mcl))


def write_families(families, output_filename):
    outputfile = open(output_filename, 'w')
    families_sorted = sorted(
        families, key=lambda mirna: (mirna.cluster, mirna.id))
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


def write_families_sort_by_size(families, output_filename, families_by_size):
    outputfile = open(output_filename, 'w')
    families_sorted = sorted(
        families, key=lambda mirna: (-families_by_size[mirna.cluster], mirna.cluster, mirna.id))
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


def write_families_mirbase_format(families, output_filename, families_by_size):
    outputfile = open(output_filename, 'w')
    families_sorted = sorted(
        families, key=lambda mirna: (-families_by_size[mirna.cluster], mirna.cluster, mirna.id))
    cluster = ""
    first_line = True
    cluster_count = 1
    for mirna in families_sorted:
        if cluster != mirna.cluster:
            cluster = mirna.cluster
            if not first_line:
                print("//", file=outputfile)
            first_line = False
            print("AC   " + "MIPF" + f"{cluster_count:07d}", file=outputfile)
            print("ID   " + cluster, file=outputfile)
        print("MI   " + mirna.ac + "  " + mirna.id, file=outputfile)


def get_clusters_by_size(families):
    families_by_size = {}
    for mirna in families:
        families_by_size[mirna.cluster] = families_by_size.get(
            mirna.cluster, 0) + 1
    return families_by_size


def write_cluster_equivalences(cluster_equivalences, filename, families):
    outputfile = open(filename, 'w')
    print("miRBase cluster" + "\t" + "Algorithm cluster", file=outputfile)
    for key, value in cluster_equivalences.items():
        for mirna in families:
            if mirna.cluster_nr == value:
                print(mirna.cluster + "\t" + key, file=outputfile)
                break


def print_stats(families):
    cluster = ""
    total_mirnas = len(families)
    mirnas_in_cluster = 0
    mirnas_per_cluster = []
    total_unclustered = 0
    for mirna in families:
        if mirna.cluster != cluster:
            if mirnas_in_cluster == 1:
                total_unclustered += 1
            elif mirnas_in_cluster > 1:
                mirnas_per_cluster.append(mirnas_in_cluster)
            cluster = mirna.cluster
            mirnas_in_cluster = 0
        mirnas_in_cluster += 1
    total_clustered = sum(mirnas_per_cluster)
    nr_clusters = len(mirnas_per_cluster)
    print("Total miRNAs: " + str(total_mirnas))
    print("Total number of 2+ clusters: " + str(nr_clusters))
    print("Total clustered: " + str(total_clustered))
    print("Total unclustered: " + str(total_unclustered))
    print("Average mirnas per 2+ cluster: " + str(total_clustered/nr_clusters))
    print("Coverage: " + str(sum(mirnas_per_cluster)/total_mirnas*100) + "%")
    print()
    return nr_clusters


def get_mirna_fasta_ids(filename):
    ids_list = []
    with open(filename) as inputfile:
        for line in inputfile.readlines():
            if line.startswith(">"):
                ids_list.append(line.split(" ")[0][1:])
    return ids_list


def get_clustal_distance_matrix(mirnas_ids_list, fasta_filename, assume_file):
    dmatrix_filename = "/media/samuacosta/Data/clust/dmatrix.mat"
    if not assume_file:
        run_clustal(fasta_filename, dmatrix_filename)
    dmatrix = pandas.read_csv(dmatrix_filename, sep=" ", engine="c",
                              skipinitialspace=True, header=0, names=mirnas_ids_list)
    print(dmatrix)
    print(dmatrix.shape)
    return dmatrix


def run_clustal(fasta_filename, clustal_filename):
    template = "clustalo -i #fasta_filename# --clustering-out=#clustal_filename# --force"
    command = template.replace("#filename#", fasta_filename).replace(
        "#clustal_filename#", clustal_filename)
    print()
    print("--- Running clustal")
    print(subprocess.getoutput(command))
    print()
    return clustal_filename


def get_matrices_from_blast(blast_filename):
    dframe_distance = pandas.read_csv(blast_filename, sep="\t", usecols=[
        0, 1, 10], header=None, names=["miRNA1", "miRNA2", "evalue"])
    # Store the miRNAs in the file to make the matrix square
    indices = set(dframe_distance['miRNA1'].unique())
    columns = set(dframe_distance['miRNA2'].unique())
    complementary_indices = list(columns-indices)
    complementary_columns = list(indices-columns)

    # Fill the missing miRNAs to get square matrix
    indices_array = []
    columns_array = []
    evalue_array = []
    if len(complementary_indices) > 0:
        indices_array.extend(complementary_indices)
        columns_array.extend(complementary_indices)
        evalue_array.extend([0] * len(complementary_indices))
    if len(complementary_columns) > 0:
        columns_array.extend(complementary_columns)
        indices_array.extend(complementary_columns)
        evalue_array.extend([0] * len(complementary_columns))
    complementary_dframe = pandas.DataFrame(
        {"miRNA1": indices_array, "miRNA2": columns_array, "evalue": evalue_array})
    dframe_distance = dframe_distance.append(complementary_dframe)
    del complementary_dframe

    # Remove duplicate pairs (leave only the one with best score)
    dframe_distance.drop_duplicates(
        subset=['miRNA1', 'miRNA2'], keep='first', inplace=True)
    # Get dataframe for similarity matrix
    # dframe_similarity = dframe_distance.copy()
    # dframe_similarity["evalue"] = dframe_similarity["evalue"].apply(
    #    get_similarity_complement)
    # Set indices for unstacking
    dframe_distance.set_index(["miRNA1", "miRNA2"], inplace=True)
    # dframe_similarity.set_index(["miRNA1", "miRNA2"], inplace=True)
    # Unstack the dataframes and get the equivalent matrices
    dframe_distance = dframe_distance.unstack()
    mirna_labels = dframe_distance.index.values.tolist()
    distance_matrix = dframe_distance.to_sparse().to_coo()
    # similarity_matrix = dframe_similarity.unstack().to_sparse().to_coo()
    print()
    return mirna_labels, distance_matrix


def get_similarity_complement(distance):
    return float(decimal.Decimal(1)-decimal.Decimal(distance))


def get_clustal_families(fasta_filename, clustal_filename, cached):
    clustal_families = []
    if not cached:
        run_clustal(fasta_filename, clustal_filename)
    with open(clustal_filename) as file:
        for line in file:
            if line != "" and line != "\n":
                mirna = Mirna()
                mirna.id = line.split(" ")[8]
                mirna.cluster_nr = line.split(" ")[1].replace(":", "")
                clustal_families.append(mirna)
    return clustal_families


def file_outputs(algorithm_families, cluster_equivalences, mirbase_families, raw_files, families_filename):
    print("--- Generating output files.")
    print()
    families_by_size = get_clusters_by_size(algorithm_families)
    if raw_files:
        equivalences_file = families_filename + "_equivalences.tsv"
        by_cluster_name_file = families_filename + "_byname.txt"
        by_cluster_size_file = families_filename + "_bysize.txt"
        clusters_only_file = families_filename + "_labelsonly.txt"
        write_cluster_equivalences(
            cluster_equivalences, equivalences_file, mirbase_families)
        write_families(algorithm_families, by_cluster_name_file)
        write_families_labels_only(algorithm_families, clusters_only_file)
        write_families_sort_by_size(
            algorithm_families, by_cluster_size_file, families_by_size)
    write_families_mirbase_format(
        algorithm_families, families_filename + ".dat", families_by_size)


def file_outputs_mirbase_families(mirbase_families, families_filename):
    print("--- Generating raw miRBase output files.")
    print()
    families_by_size = get_clusters_by_size(mirbase_families)
    by_cluster_name_file = families_filename + "_byname.txt"
    by_cluster_size_file = families_filename + "_bysize.txt"
    clusters_only_file = families_filename + "_labelsonly.txt"
    write_families(mirbase_families, by_cluster_name_file)
    write_families_labels_only(mirbase_families, clusters_only_file)
    write_families_sort_by_size(
        mirbase_families, by_cluster_size_file, families_by_size)


# Entry
if __name__ == "__main__":
    main()
