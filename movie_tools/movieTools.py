import pandas as pd
import itertools
from scipy.sparse import coo_matrix

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





# Sample Call:
# (dir_hitRate, dir_hit_dict, dir_hit_counts, dir_total_movies) = running_hitRate(dir_binary, data['TARGET'], Laplace=True)

def running_hitRate(B, target, Laplace=False, first_time_default=0.5, DT=np.float32):
    """
    INPUTS:
    B = sparse CSR matrix word count matrix from binary_vectorizer
        B MUST BE SORTED IN ASCENDING TIME ORDER! Oldest (top) -> Newest (bottom)
    target = vector of 1 or 0 (whether the target variable is True or False)
             must have same length as # of movies
             also could be numeric ROI to find average...
    Laplace = True/False (whether or not to do Laplace smoothing)
    first_time_default = value to assign to first-time directors
                         TODO: check if Laplace default is 0.5 or 1/3???
    DT = default datatype (trying to save memory)
    
    RETURNS:
        (out_matrix, final_hit_dict, hits, totals)
        out_matrix: CSR sparse formatted matrix with running hit ratio
        final_hit_dict: Dictionary mapping {director_ID: final_hit_ratio} eg. {'nm234u23': 0.5}
        hits: Numpy Array of final hit count
        totals: Numpy Array of total movie count per director
    """
    
    A = B.tocoo()    # convert to COO sparse format to enable iteration
    
    first_time_hits = 0.0   # keep count of first time hits
    first_time_total = 0.0  # total count of first-timer directors (should match # of unique directors)
    
    if Laplace:
        # Defaults for Laplace Smoothing:
        totals = 2.0 * np.ones( (1, A.shape[1]), dtype=DT).flatten()
        hits = np.ones_like(totals).flatten()
        first_time_hits += 1
        first_time_total += 2
        #first_time_default = 1.0 / 3.0; # CHECK THIS!
    else:
        # just simple zeros
        totals = np.zeros( (1, A.shape[1]), dtype=DT).flatten()
        hits = np.zeros_like(totals).flatten()
        
    # make empty row/column containers for output:
        # could probably just make copies, but not 100% sure about iteration order, not worth the risk
    out_rows = np.zeros_like(A.row)
    out_cols = np.zeros_like(A.col)
    out_data = np.zeros(out_cols.shape, dtype=DT)
    
    #nz_count = 0
    loop_count = 0
    """ 
    i = movie index (row number)
    j = director index (column number) - also same index for totals, hits vectors
    v = value of A at position (i,j)
    """
    for i,j,v in itertools.izip(A.row, A.col, A.data):
        #print "A(%d, %d) = %.2f" % (i, j, v)
        
        #if v == 1:   # for directors, should be only one "1" per row
            #this is redundant- only loops over non-zero values, which are all 1 in our case...
            
        # update entry in output matrix:
        out_rows[loop_count] = i 
        out_cols[loop_count] = j   

        if totals[j] == 0: 
        # above is INCORRECT as far as accurate Laplace est. for first-timers (b/c it's init to 2)
        # BUT it works for correctly assigning default when not-Laplace, and using Laplace 0.5 if Laplace=True.
            # corrected version (not necessary):
        #if (totals[j] == 0) or (Laplace and totals[j] == 2):   # first time director
            out_data[loop_count] = first_time_default
            
            # track first-timer statistics:
            first_time_total += 1
            first_time_hits += target[i]
        else:
            # if not a first-timer, assign most recent hit ratio for director[j]
            out_data[loop_count] = hits[j] / totals[j]
            
        # update totals & hits:
        # This MUST happen AFTER assigning a value- can't use "future" knowledge eg. actual value of target
        totals[j] += 1.0
        hits[j] += target[i] # this might be better for incorporating ROI 
        
        #else:
        #    nz_count += 1
        loop_count += 1
        
    # end loop 
            
    #print "NNZ: %d" % nz_count
    print "loop count: %d" % loop_count
    Asize = np.prod(A.shape)
    print "# elements in A: %d" % Asize
    print "pct. of matrix looped over: %.8f" % (float(loop_count) / Asize)
    print
                
    # Error Checking:
    if not Laplace:
        assert np.allclose(totals, A.sum(axis=0)), '"totals" should be equal to row sum of A'
        print hits.sum()
        print target.sum()
        assert hits.sum() == target.sum(), '"Hits sum/Target sum mismatch"'
    
    totals_sum = totals.sum()
    hits_sum = hits.sum()
    print "Totals sum: %.2f" % totals_sum 
    print "Hits sum: %.2f" % hits_sum 
    print "P(hit): %.5f" % (hits_sum / totals_sum)
    if Laplace:
        print "IGNORE THIS (wrong counts with Laplace):"
    print "P(first-timer hit): %.5f" % (first_time_hits / first_time_total)
    print "First Timer Hits: " + str(first_time_hits)
    print "First Timer Total: " + str(first_time_total)
    
    # create new matrix and convert to CSR sparse format (for sklearn?)
    out_matrix = coo_matrix((out_data, (out_rows, out_cols)), shape=A.shape).tocsr()
    
    most_recent_vec = np.divide(hits, totals) # most recent hit ratio- to be mapped to a dictionary later
    # create dictionary mapping {directorID: finalHitRate}: (using dict comprehension)
    final_hit_dict = {i: hr for i, hr in itertools.izip(binary_vectorizer.get_feature_names(), most_recent_vec) }
    
    return (out_matrix, final_hit_dict, hits, totals)


# From Katrina's original ipynb file:

# Function splitting input data into training and testing sets
def time_holdout(data, cutoff_year, window=0):
    # data = sorted input data
    # cutoff_year is the first year in the test set
    # window is the number of years of training data to include; 0 by default => ALL data is used
    data['year'] = pd.to_datetime(data['ReleaseDate']).dt.year
    
    if window == 0:
        start_year = min(data['year'])
    else:
        start_year = cutoff_year - window
        
    training_set = data[(data['year']>=start_year) & (data['year']<cutoff_year)]
    test_set = data[data['year']>=cutoff_year]
    
    training_set = training_set.drop('year', 1)
    test_set = test_set.drop('year', 1)
    
    return training_set, test_set









def runSingleROCtest(X_train, Y_train, X_test, Y_test, title_str):
    
    model_types = ["Decision Tree", "Logistic Regression"]
    
    for mtype in model_types:
        print mtype
        if mtype == "Decision Tree":
            model = DecisionTreeClassifier(criterion="entropy") #, min_samples_split = 50) # TODO: figure out best MSS (min sample split or maxDepth)
        else:    
            model = LogisticRegression()   # TODO: find Best "C" or L1/L2 penalty function
        model.fit(X_train, Y_train)
        
        # Get the predicted value and the probability of Y_test records being = 1
        Y_test_predicted = model.predict(X_test)
        Y_test_probability_1 = model.predict_proba(X_test)[:, 1]
    
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_probability_1)
        
        auc = metrics.roc_auc_score(Y_test, Y_test_probability_1) 
        
        
        plt.plot(fpr, tpr, label=mtype + " (AUC = " + str(round(auc, 2)) + ")")    
        
    plt.title(title_str)
    plt.xlabel('False Positive Rate (fpr)')
    plt.xlabel('True Positive Rate (tpr)')
    plt.plot([0,1], [0,1], 'k--', label="Random")
    plt.legend(loc=2)


def LR_C_value_curve(X_train, Y_train, X_test, Y_test,
                     C_vals=[0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 10.0, 16.0, 20.0]):
    
    auc_vals = []
    best_auc = 0.0
    best_c = -1
   
    for c in C_vals:
        model = LogisticRegression(C = c)   # TODO: find Best "C" or L1/L2 penalty function
        model.fit(X_train, Y_train)
        
        # Get the predicted value and the probability of Y_test records being = 1
        Y_test_predicted = model.predict(X_test)
        Y_test_probability_1 = model.predict_proba(X_test)[:, 1]
    
        # this is for ROC curve
        # fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_probability_1)
        
        auc = metrics.roc_auc_score(Y_test, Y_test_probability_1) 
        auc_vals.append(auc)
        
        if auc > best_auc:
            best_c = c
            best_auc = auc
        print 'C: %.4f, AUC: %.2f' % (c, auc)
        
    print
    print "Best C Value: " + str(best_c)
    print "Best AUC: %.5f" % best_auc
        
    plt.plot(C_vals, auc_vals, '-^')
    plt.title('AUC vs. C value (complexity curve)')
    plt.xlabel('C value')
    plt.ylabel('AUC value')




def DT_MSS_value_curve(X_train, Y_train, X_test, Y_test,
                     mss_vals=[1, 2, 4, 8, 16, 24, 32, 64, 100, 200]):
    
    auc_vals = []
    best_auc = 0.0
    best_mss = -1
   
    for mss in mss_vals:
        model = DecisionTreeClassifier(criterion="entropy", min_samples_split = mss)
        model.fit(X_train, Y_train)
        
        # Get the predicted value and the probability of Y_test records being = 1
        Y_test_predicted = model.predict(X_test)
        Y_test_probability_1 = model.predict_proba(X_test)[:, 1]
    
        auc = metrics.roc_auc_score(Y_test, Y_test_probability_1) 
        auc_vals.append(auc)
        
        if auc > best_auc:
            best_mss = mss
            best_auc = auc
        print 'MSS: %d, AUC: %.5f' % (mss, auc)
        
    print
    print "Best MSS Value: %d" % best_mss
    print "Best AUC: %.5f" % best_auc
        
    plt.plot(mss_vals, auc_vals, '-^')
    plt.title('AUC vs. MSS value (complexity curve)')
    plt.xlabel('MSS value')
    plt.ylabel('AUC value')



"""
# SAMPLE CALL:
runSingleROCtest_wParams(CX_train, CY_train, CX_test, CY_test,
                         title_str="Best DT/LR for HitRate SINGLE COLUMN",
                         model_types=["Decision Tree", "Logistic Regression"],
                         params=[170, 0.01])
"""

def runSingleROCtest_wParams(X_train, Y_train, X_test, Y_test, title_str, model_types=["Decision Tree", "Logistic Regression"], params=None):
    
    #model_types = ["Decision Tree", "Logistic Regression"]
    if params is not None:
        assert len(model_types) == len(params), 'If Params are specified, must be one per model type'
    
    for mtype, p in zip(model_types, params):
        print mtype 
        print p
        print
        
        if mtype == "Decision Tree":
            if params is not None:
                model = DecisionTreeClassifier(criterion="entropy", min_samples_split=p) 
            else: # use default
                model = DecisionTreeClassifier(criterion="entropy")
        else:    
            if params is not None:
                model = LogisticRegression(C = p)   # TODO: find Best "C" or L1/L2 penalty function
            else:
                model = LogisticRegression()   # TODO: find Best "C" or L1/L2 penalty function
        model.fit(X_train, Y_train)
        
        # Get the predicted value and the probability of Y_test records being = 1
        Y_test_predicted = model.predict(X_test)
        Y_test_probability_1 = model.predict_proba(X_test)[:, 1]
    
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_probability_1)
        
        auc = metrics.roc_auc_score(Y_test, Y_test_probability_1) 
        
        
        plt.plot(fpr, tpr, label=mtype + " (AUC = " + str(round(auc, 2)) + ")")    
        
    plt.title(title_str)
    plt.xlabel('False Positive Rate (fpr)')
    plt.xlabel('True Positive Rate (tpr)')
    plt.plot([0,1], [0,1], 'k--', label="Random")
    plt.legend(loc=2)




