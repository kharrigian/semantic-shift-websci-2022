
"""
Extract vocabulary from an embedding file and save it to disk
"""

###################
### Imports
###################

## Standard Libary
import os
import argparse

###################
### Functions
###################

def parse_command_line():
    """
    
    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("embedding_filename",
                            type=str)
    args = parser.parse_args()
    return args

def load_embeddings(filename):
    """
    
    """
    ## Check for File
    if not os.path.exists(filename):
        raise FileNotFoundError("Embedding file does not exist.")
    ## Output Directory
    output_dir = "{}/".format(os.path.dirname(filename)).replace("//","/")
    ## Identify Dimension
    dim = int(os.path.basename(filename).split("embeddings.")[1].split("d.txt")[0])
    ## Load Vocabulary
    vocabulary = []
    with open(filename,"r") as the_file:
        for line in the_file:
            line_term = " ".join(line.split()[:-dim])
            vocabulary.append(line_term)
    return vocabulary, output_dir

def save_vocabulary(vocabulary,
                    output_dir):
    """
    
    """
    ## Identify Filename
    vfile = f"{output_dir}/vocabulary.txt".replace("//","/")
    ## Dump
    with open(vfile,"w") as the_file:
        for term in vocabulary:
            the_file.write(f"{term}\n")
    return vfile

def main():
    """
    
    """
    ## Parse Command Line
    args = parse_command_line()
    ## Load Vocabulary
    vocabulary, output_dir = load_embeddings(args.embedding_filename)
    ## Cache Vocabulary
    vocabulary_file = save_vocabulary(vocabulary, output_dir)
    ## Alert User
    print(f"Vocabulary Saved to: {vocabulary_file}")

#####################
### Execute
#####################

if __name__ == "__main__":
    _ = main()