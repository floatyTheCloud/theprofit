import pandas as pd

def checkIDs(L):
    # L = list of Pandas dataframes, with 'index' set to IMDB Movie ID ***THIS IS IMPORTANT!***
    # returns True/False if all index lists match

    numel = len(L[0]) # length of first list
    checklist = L[0].index.tolist() 
    good = True
    for l in L[1:]:  # check 2nd and further lists against first list (0)
        if not l.index.tolist() == checklist:
            good = False
            #print "" # TODO: Figure out how to get the name of the list to print
            if len(l) != numel:
                print "Error: Lists are different lengths: %d, %d" % ( len(bar_info), len(dir_info) )        
            break
    return good
