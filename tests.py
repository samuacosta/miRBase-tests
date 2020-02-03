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

__author__ = "Samuel Acosta"
__email__ = "samuel.acostamelgarejo@postgrad.manchester.ac.uk"
__version__ = "1.0.0"

# Parametrics


def main():
    '''Main method'''
    try:
        print("----- Running")

        # Read command line parameters
        args = read_parameters()
        blast_filename = args.file  # data/hairpin_m8.cblast
        # Call bash commands (mcl)
        # execute_bash(blast_filename)
        # write_families()
        # Generates another file, sorted by cluster ID (better for comparison)
        produce_sorted_file("miFamMcl.dat")
        produce_sorted_file("data/miFam.dat")

        print("----- Finished succesfully")
    except Exception as e:
        print("An error occurred during execution:")
        print(e)
        raise


def read_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--file",
                        nargs='*',
                        help="The name of the optional output file. [Default=hairpin_m8.cblast]",
                        default="data/hairpin_m8.cblast"
                        )
    return(parser.parse_args())


def execute_bash(blast_filename):
    command = "mcxdeblast --m9 --line-mode=abc --out=- " + \
        blast_filename + " | mcl - --abc -o out.mcl"
    print(subprocess.getoutput(command))


def write_families():
    aliases = load_aliases("data/aliases.txt")
    outputfile = open("miFamMcl.dat", 'w')
    familynr = 0
    with open('out.mcl') as inputfile:
        for line in inputfile:
            mirnas = line.split("\t")
            # Removing 1-mirna clusters
            if len(mirnas) > 1:
                familynr += 1
                print("AC   MIPF" + str(familynr).zfill(7), file=outputfile)
                mirnas_subids = [
                    mirna[mirna.index("-")+1:] for mirna in mirnas]
                cluster_id = get_cluster_id(mirnas_subids)
                print("ID   " + cluster_id, file=outputfile)
                for mirna in mirnas:
                    mirna = mirna.strip("\n")
                    print("MI   " + aliases.get(mirna) +
                          "  " + mirna, file=outputfile)
                print("//", file=outputfile)


def load_aliases(aliases_filename):
    aliases = {}
    with open(aliases_filename) as file:
        for line in file:
            (alias, mirnas) = line.split("\t")
            for mirna in mirnas.split(";"):
                if mirna != "" and mirna != "\n":
                    aliases[mirna] = alias
    return aliases


def get_cluster_id(mirnas_subids):
    prefix_ranking = sorted([(sum([w.startswith(prefix)
                                   for w in mirnas_subids]), prefix) for prefix in mirnas_subids])[::-1]
    return prefix_ranking[0][1].strip("\n")


def produce_sorted_file(filename):
    families_list_sorted = get_family_list_sorted(filename)
    outputfile = open(filename + ".sorted", 'w')
    for family in families_list_sorted:
        print(family["cluster_id"], file=outputfile)
        mirnas = family["mirnas"]
        for mirna in mirnas:
            print(mirna, file=outputfile)
        print("//", file=outputfile)


def get_family_list_sorted(filename):
    families_list = []
    with open(filename) as inputfile:
        family = {}
        cluster_ac = ""
        cluster_id = ""
        mirnas = []
        first_line = True
        for line in inputfile:
            if line.startswith("AC") or line.startswith("//") or line == "":  # ignore
                continue
            elif line.startswith("ID"):
                if not first_line:
                    cluster_id = cluster_id.strip("\n")
                    family["cluster_id"] = cluster_id
                    mirnas.sort()
                    family["mirnas"] = mirnas
                    families_list.append(family.copy())
                    mirnas = []
                else:
                    first_line = False
                cluster_id = line[5:]
            else:
                mirna = line.split("  ")[2].strip("\n")
                mirnas.append(mirna)

    families_list_sorted = sorted(
        families_list, key=operator.itemgetter('cluster_id'))
    return families_list_sorted


# Entry
if __name__ == "__main__":
    main()
