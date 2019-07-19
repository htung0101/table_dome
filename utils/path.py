
import os

def makedir(path):

    # check if its upper path is there
    parent_path = "/".join(path.split("/")[:-2])
    if os.path.exists(parent_path):
        KeyError(f"{parent_path} does not exists")
 
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        KeyError(f"{path} already exists. Refuse to overwrite!")

