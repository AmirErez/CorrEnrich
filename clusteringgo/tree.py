"""
Module for building and handling the Gene Ontology (GO) tree structure.
"""
import os
import requests
from collections import defaultdict, deque
from typing import Dict, Set, Any, Tuple
import numpy as np

import wget
from anytree import NodeMixin, PostOrderIter
from goatools import obo_parser
from biomart import BiomartServer


class GeneNode(NodeMixin):
    """Represents a node in the GO tree."""

    def __init__(self, go_id, level, name, go_obj, parents=None, children=None):
        super().__init__()
        self.go_id = go_id
        self.level = level
        self.name = name
        category = list(get_ancestor(go_obj))
        self.category = category[0].name if len(category) else "biological_process"
        self.parents = parents if parents is not None else []
        self.children = children if children is not None else []
        self.gene_set = set()
        self.pearson_corr = None
        self.spearman_corr = None
        self.dist = np.inf

    def __repr__(self):
        return self.go_id


# def get_go(data_dir=".", download_anyway=False):
#     """Downloads the go-basic.obo file if not present."""
#     go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
#     os.makedirs(data_dir, exist_ok=True)
#     obo_path = os.path.join(data_dir, 'go-basic.obo')
#     if not os.path.isfile(obo_path) or download_anyway:
#         wget.download(go_obo_url, obo_path)
#     return obo_path
def get_go(data_dir=".", download_anyway=False):
    """
    Downloads the go-basic.obo file if not present using the requests library.
    """
    # This PURL will be correctly followed by requests
    go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'

    os.makedirs(data_dir, exist_ok=True)
    obo_path = os.path.join(data_dir, 'go-basic.obo')

    if not os.path.isfile(obo_path) or download_anyway:
        print(f"Downloading {go_obo_url} to {obo_path}...")
        try:
            # allow_redirects=True is the default, but good to be explicit
            r = requests.get(go_obo_url, allow_redirects=True)

            # This will raise an error if the download failed (e.g., 404, 500)
            r.raise_for_status()

            # Write the content to the file in binary mode
            with open(obo_path, 'wb') as f:
                f.write(r.content)
            print("\nDownload complete.")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            return None

    return obo_path


def get_ancestor(go_term):
    """Finds the top-level ancestor of a GO term."""
    last = set()
    to_check = {go_term}
    while to_check:
        term = to_check.pop()
        if not term.parents:
            last.add(term)
        for parent in term.parents:
            if parent.id == "GO:0008150":  # biological_process
                last.add(term)
            else:
                to_check.add(parent)
    return last


def build_genomic_tree(biological_processes: Any, go: Dict) -> Tuple[GeneNode, int]:
    """
    Builds a tree structure from the GO DAG using BFS.
    """
    visited: Set[str] = set()
    root = GeneNode(go_id=biological_processes.id, level=biological_processes.level,
                    name=biological_processes.name, go_obj=biological_processes)
    to_visit = deque([root])
    id_to_node: Dict[str, GeneNode] = {biological_processes.id: root}
    nodes = 0

    while to_visit:
        current = to_visit.popleft()
        if current.go_id in visited:
            continue

        visited.add(current.go_id)
        nodes += 1

        if current.go_id in go:
            children_nodes = []
            for child in go[current.go_id].children:
                if child.id not in id_to_node:
                    temp_node = GeneNode(go_id=child.id, level=child.level, name=child.name, go_obj=child)
                    id_to_node[child.id] = temp_node
                    to_visit.append(temp_node)
                else:
                    temp_node = id_to_node[child.id]
                children_nodes.append(temp_node)
                temp_node.parents.append(current)
            current.children = children_nodes

    return root, nodes


def get_go_to_ensmusg():
    """Fetches GO to Ensembl gene mappings from BioMart."""
    server = BiomartServer("http://www.ensembl.org/biomart")
    mart = server.datasets['mmusculus_gene_ensembl']
    attributes = ['ensembl_gene_id', 'go_id']
    filters = {'go_parent_term': 'GO:0008150'}

    response = mart.search({'filters': filters, 'attributes': attributes})

    go_to_ensmusg = defaultdict(set)
    for line in response.iter_lines():
        decoded_line = line.decode('utf-8')
        if "\t" in decoded_line:
            ensembl_gene_id, go_id = decoded_line.split("\t")
            if go_id:
                go_to_ensmusg[go_id].add(ensembl_gene_id)
    return go_to_ensmusg


def add_genes_ids(root: Any, go_to_ensmbl_dict: Dict[str, Set[str]]):
    """Adds gene sets to each node in the GO tree."""
    for node in PostOrderIter(root):
        node_genes = go_to_ensmbl_dict.get(node.go_id, set())
        if node_genes:
            node.gene_set.update(node_genes)
    return root


def get_go_to_ensmusg():
    from biomart import BiomartServer

    # Connect to the BioMart server
    server = BiomartServer("http://www.ensembl.org/biomart")

    # Choose the Ensembl database
    mart = server.datasets['mmusculus_gene_ensembl']

    # Define the attributes you want to retrieve
    attributes = [
        'ensembl_gene_id',
        'go_id'
    ]
    filters = {
        'go_parent_term': 'GO:0008150'  # This is the root term for Biological Process
    }

    # Query BioMart
    response = mart.search({
        'filters': filters,
        'attributes': attributes
    })

    # Parse the response
    go_to_ensmusg = defaultdict(set)
    for line in response.iter_lines():
        decoded_line = line.decode('utf-8')
        ensembl_gene_id, go_id = decoded_line.split("\t")
        if go_id:
            go_to_ensmusg[go_id].add(ensembl_gene_id)
    return go_to_ensmusg


def build_tree(data_dir=".", download=False):
    """High-level function to build the complete GO tree with genes."""
    go_dag = obo_parser.GODag(get_go(data_dir, download_anyway=download))

    root_node_obj = go_dag.get('GO:0008150')  # biological_process
    if not root_node_obj:
        raise ValueError("Could not find root GO term 'GO:0008150' in OBO file.")

    tree, _ = build_genomic_tree(root_node_obj, go_dag)

    go_to_ensmbl_dict = get_go_to_ensmusg()
    tree_with_genes = add_genes_ids(tree, go_to_ensmbl_dict)

    return tree_with_genes, len(go_dag)
