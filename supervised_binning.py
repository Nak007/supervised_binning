'''
Available methods are the followings:
[1] batch_binning 
[2] woe_binning
[3] evaluate_bins
[4] plot_woe
[5] woe_transform

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 08-09-2020
'''
from inspect import signature
import pandas as pd, numpy as np, time, inspect, numbers, random
from warnings import warn
import matplotlib.pylab as plt
from scipy.stats import (spearmanr, pearsonr, sem, t, chi2, kendalltau)
import scipy.stats as st, ipywidgets as widgets
from IPython.display import HTML, display
from sklearn.linear_model import LogisticRegression
from itertools import product

__all__ = ['batch_binning','woe_binning','evaluate_bins',
           'plot_woe','woe_transform']

def _get_argument(func):
    
    '''
    Parameters
    ----------
    func : function object
    
    Returns
    -------
    - `array` of parameter names in required 
       arguments.
       
    - `dict` of parameter names in positional 
      arguments and their default value.
    '''
    # Get all parameters from `func`.
    params = signature(func).parameters.items()
        
    # Internal functions for `parameters`.
    is_empty = lambda p : p.default==inspect._empty
    to_dict = lambda p : (p[1].name, p[1].default)
        
    # Take argument(s) from `func`.
    args = np.array([k[1].name for k in params if is_empty(k[1])])
    kwgs = dict([to_dict(k) for k in params if not is_empty(k[1])])
  
    return None if len(args)==0 else args[args!='self'], kwgs

def _is_array_like(X):
    '''
    Returns whether the input is array-like.
    '''
    return (hasattr(X, 'shape') or hasattr(X, '__array__'))

def _to_DataFrame(X):
    '''
    If `X` is not `pd.DataFrame`, column(s) will be
    automatically created with "Unnamed_XX" format.
    
    Parameters
    ----------
    X : array-like or `pd.DataFrame` object
    
    Returns
    -------
    `pd.DataFrame` object.
    '''
    digit = lambda n : int(np.ceil(np.log(n)/np.log(10)))             
    if _is_array_like(X)==False:
        raise TypeError('Data must be array-like.')
    elif isinstance(X, (pd.Series, pd.DataFrame)): 
        return pd.DataFrame(X)
    elif isinstance(X, np.ndarray) & (len(X.shape)==1):
        return pd.DataFrame(X.reshape(-1,1),columns=['Unnamed'])
    elif isinstance(X, np.ndarray) & (len(X.shape)==2):
        z = digit(X.shape[1])
        columns = ['Unnamed_{}'.format(str(n).zfill(z)) 
                   for n in range(X.shape[1])]
        return pd.DataFrame(X,columns=columns)

def _combination_(param_grid):
    '''
    Create combination from all attributes.

    Parameters
    ----------
    param_grid : `dict` object
    \t Dictionary with parameters names (`str`) as keys 
    \t and lists of parameter settings to try as values, 
    \t or a single value (`str`,`float` or `int`).

    Returns
    -------
    attrs : `dict` object
    \t Dictionary with combination number (`int`) as
    \t keys and dictionary of attributes.
    '''
    attrs = [list(product(*[[key],param_grid[key]])) 
             if isinstance(param_grid[key],list) 
             else [(key,param_grid[key])] 
             for key in param_grid.keys()]
    attrs = dict([(n,dict(d)) for n,d in 
                  enumerate(list(product(*attrs)),1)])
    return attrs

class batch_binning:
    '''
    Exhaustive search over specified parameter values for 
    woe_binning class.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_breast_cancer as data
    
    >>> X, y = data(return_X_y=True)
    >>> columns = [s.replace(' ','_') 
    ...            for s in data().feature_names]
    >>> X = pd.DataFrame(X, columns=columns)
    
    # Set `param_grid`.
    >>> params = {'param_grid':{'method':['gini','mono'], 
    ...                         'equal_bin':[None,4,5], 
    ...                         'n_step':[10,30],
    ...                         'sub_step':[5,15],
    ...                         'n_order':[2,None],
    ...                         'random_state':99}}
    >>> test = batch_binning(**params)
    UserWarning: There are 12 combinations.
    
    >>> test.fit(y, X)
    
    # Measuring metrics of each iteration.
    >>> pd.DataFrame(test.data['log']).head()
    
    # Result of all binnings.
    >>> pd.DataFrame(test.data['data']).head()
    
    # Binning method that scores the highest for each variable.
    >>> pd.DataFrame(test.data['best_bins']).head()
    
    # Parameter grid (dictionary).
    >>> test.data['combinations']
    '''
    def __init__(self, **params):
        
        '''
        Parameters
        ----------
        **params : dictionary of properties, optional
        \t `params` is used to specify or override 
        \t properties of 
            
            use_equal_bin : `bool`, optional, default: True
            \t `use_equal_bin` is not applicable under
            \t `mono`, and `chi`. If `False`, `equal_bin` 
            \t defaults to `None` and becomes inactive. 
            \t Since `equal_bin` tends to generate more bins 
            \t than `np.nanpercentile`, so it would be time 
            \t consuming when top-down approach (e.g. `gini`) 
            \t is selected, particularly, when `batch_binning` 
            \t is implemented with large dataset and multiple
            \t variables. 

            param_grid : `dict` object
            \t Initial parameters for `woe_binning` class.
            \t (see `woe_binning.__init__.__doc__`)
            
            evaldict : `dict` object
            \t Parameters for `evaluate_bins.fit` method.
            \t (see `evaluate_bins.fit.__doc__`)
            
            importance : `dict` object
            \t Parameters for `importance_order` method.
            \t (see `self.importance_order.__doc__`)

        Returns
        -------
        self.combinations : `dict` object
        \t Combinations resulted from `param_grid`.
        
        Notes
        -----
        `equal_bin` can be a list of `int` e.g. [None, 3, 6].
        If `None`, the algorithm will implement equal binning
        based on `n_step`.
        '''
        # Get `use_equal_bin` arguments.
        if params.get('use_equal_bin') is not None:
            use_equal_bin = params.get('use_equal_bin')
        else: use_equal_bin = True
        
        # Get `param_grid` arguments.
        param_grid = _get_argument(woe_binning)[1]
        if params.get('param_grid') is not None:
            param_grid = {**param_grid,
                          **params.get('param_grid')}
            
        # Get `evaldict` arguments.
        if params.get('evaldict') is not None:
            self.evaldict = params.get('evaldict')
        else:self.evaldict = _get_argument(evaluate_bins)[1]
        
        # Get `importance` arguments.
        self.importance = _get_argument(self.importance_order)[1] 
        if params.get('importance') is not None:
            self.importance = {**self.importance,
                               **params.get('importance')}
           
        # Create combination of attributes.
        param_grid = _combination_(param_grid)
        
        # Default arguments for each method.
        def default(method):
            if method == 'chi':
                return {'trend':'auto', 'n_order':1, 
                        'random_state':None, 
                        'sub_step':10, 'p_value':0.05}
            elif method == 'mono': 
                return {'trend':'auto', 'n_order':1, 
                        'random_state':None, 
                        'sub_step':10, 'chi_alpha':0.05}
            else: return {'n_step':20, 'chi_alpha':0.05, 
                          'p_value':0.05}

        # ==================================================
        # Create dictionary of parameter grid. `n_step`
        # would no longer matter when 'equal_bin' equals to 
        # some integer. Thus, such `n_step` is changed to 1,
        # which allows algorithm to remove duplicates.
        # --------------------------------------------------
        # ** woe_binning's arguments **
        # (trend='auto', n_order=1, random_state=None, 
        #  equal_bin=None, n_step=20, sub_step=10, 
        #  method='iv', min_pct=0.05, min_labels=(0.01,0.01), 
        #  chi_alpha=0.05, p_value=0.05)
        # --------------------------------------------------
        iterations = dict()
        for key,value in param_grid.items(): 
            # Update default arguments.
            value = {**value,**default(value['method'])}
        # -------------------------------------------------- 
            # When `n_order` is `int`, this overrides
            # random function, where it generates random
            # order. Therefore, `random_state` is no longer
            # necessary and shall be assigned to `None`.
            if value['n_order'] is not None: 
                value['random_state'] = None
        # --------------------------------------------------
            # Override `equal_bin`.
            if use_equal_bin==False:
                if value['method'] not in ['mono','chi']:
                    value['equal_bin'] = None
        # --------------------------------------------------           
            # When `equal_bin` is `int`, it overrides the
            # binning function that relates to `n_step`, and
            # `sub_step`. Hence, setting such parameters to
            # default values helps eliminate duplicates.        
            if value['equal_bin'] is not None:
                value['n_step'] = 20
                value['sub_step'] = 10
        # --------------------------------------------------
            if value not in iterations.values():
                iterations[len(iterations)+1] = value
        # ==================================================
        
        self.combinations = iterations
        print("There are {:,} combinations.".format(len(iterations)))
        
        # Columns for `log` and `data`.
        self.columns = {'log' :['round', 'method', 'variable', 
                                'model_bin', 'iv', 'correlation', 
                                'intercept', 'beta', 'combination'],
                        'data':['round', 'variable', 'min', 'max', 
                                'bin', 'non_events', 'events', 
                                'pct_nonevents', 'pct_events', 
                                'woe', 'iv']}
        
    def fit(self, y, X):
        '''
        Fit model.
        
        Parameters
        ----------
        y : array-like or `pd.Series` object
        \t Target values (binary).
        
        X : `pd.DataFrame` object
        \t Training data.
        
        Returns
        -------
        self.data : dictionary of `pd.DataFrame` objects
        \t keys of `self.data` are;
        \t 'log'  : measuring metrics of each iteration.
        \t 'data' : binning results.
        \t 'best_bins' : optimal binning method for each variable.
        \t 'combinations' : combinations resulted from `param_grid`.
        
        Attributes
        ----------
        self.missing : `list` of `str`
        \t List of variable(s) that contains nothing.

        self.constant : `list` of `str`
        \t List of variable(s), apart from missing, contains 
        \t only single value (like a constant).
        '''
        # Create updating widgets.
        t1, t2, t3, strfmt = self._create_widgets() 
        
        # Convert `X` to `pd.DataFrame`. 
        X = self.__to_df(X)
        
        # Combine variable with combination of parameters.
        combinations = [self.combinations[key] for key 
                        in self.combinations.keys()]
        iterations = list(product(*[list(X),combinations]))
        n_round = len(iterations)
        n_combi = len(combinations)

        # Initialize variables.
        data = pd.DataFrame(columns=self.columns['data'])
        self.data = dict(log=None, data=None, best_bins=None)
        start, log = time.time(), list()

        # n_iter[0] keeps variable name, while n_iter[1] 
        # contains all arguments used in `woe_binning`.
        for n,n_iter in enumerate(iterations,1):
      
            # Update progress bar.
            t = (n_iter[0], np.where(n%n_combi==0, n_combi, n%n_combi))
            t1.value = 'Variable : %s, Combination : %s' % t
            t2.value = self._progress_pct(n, n_round)
            t3.value = strfmt(self._eta(start, n, n_round),
                              n_iter[1]['method'])

            # `WOE` binning.
            woe = woe_binning(**n_iter[1])
            woe.fit(y, X[n_iter[0]])
            
            # Evaluate bins.
            eval_bins = evaluate_bins(**self.evaldict)
            eval_bins.fit(y, X[n_iter[0]], bins=woe.bin_edges)
            
            # Iteration logs.
            log.append([n, n_iter[1]['method'], 
                        n_iter[0], 
                        len(eval_bins.woe_df)-1, 
                        eval_bins.iv, 
                        eval_bins.rho, 
                        eval_bins.intercept_['delta'],
                        eval_bins.intercept_['beta'],
                        int(t[1])])

            # Binning results
            a = eval_bins.woe_df.copy(); a['round'] = n
            data = data.append(a, ignore_index=True, sort=False)
        
        # Store `log`, `data`, `best_bins`, and `combinations`.
        log = pd.DataFrame(log, columns=self.columns['log'])
        self.data['log'] = log.to_dict(orient='list')
        self.data['data'] = data.to_dict(orient='list')
        self.data['best_bins'] = self.importance_order(**self.importance)
        self.data['combinations'] = self.combinations
        
        # Elapsed-Time and number of iterationsß.
        elapsed_time = self._sec_to_hms(time.time()-start)
        text = 'Complete . . . , Elapsed-Time : {}, Iterations : {:,}'
        t1.value = text.format(elapsed_time,len(iterations))
   
    def importance_order(self, iv_imp=0.5, min_iv=0.1, min_corr=0.5, 
                         max_tol=0.1):
        
        '''
        Before applying importance weight, each iteration
        must pass all following criteria;
        
        [1] exceeds minimum Information Value (`min_iv`) or
            not equal to `np.nan` (missing).
        [2] exceeds minimum absolute correlation (`min_corr`).
        [3] remain within maximum tolerance for intercept
            (`max_tol`).

        Subsequently, importance weight is applied and its 
        score is expressed as follows;
        
                score(n) = [w]*iv(n) + [1-w]*|ρ(n)|
        
        n : nth round (iteration).
        w : importance weight (`iv_imp`).
        ρ : Pearson correlation coefficient.

        If there is more than one output (per variable) left, 
        the earliest `round` is selected.
        
        Parameters
        ----------    
        iv_imp : `float`, optional, default: 0.5
        \t Information Value (IV) importance weight
        \t (0 ≤ `iv_imp` ≤ 1). 

        min_iv : `float`, optional, default: 0.1 
        \t Minimum acceptable Infomation Value. This 
        \t value is between [0, np.inf].

        min_corr : `float`, optional, default: 0.5
        \t Minimum acceptable absolute correlation. This 
        \t value must be between [0, 1].

        max_tol : `float`, optional, default: 0.1
        \t Maximum tolerance of difference between model  
        \t intercept and log(event/nonevent). This 
        \t difference is obtained from `evaluate_bin` 
        \t class, where `max_tol` is 
        \t |model.intercept - intercept| / intercept.
        
        Returns
        -------
        dictionary object.
        '''
        data = pd.DataFrame(self.data['log'].copy())
        data = data.loc[~data['iv'].isna()]
        
        # Set conditions
        cond1 = (data['iv']>=min_iv) 
        cond2 = (abs(data['correlation'])>=min_corr)
        cond3 = (data['intercept']<=max_tol)
        data = data.loc[cond1 & cond2 & cond3]

        # Weighted score between `correlation` and `iv`.
        corr = abs(data['correlation'])*(1-iv_imp) 
        data['score'] = data['iv']*iv_imp + corr

        # Select combination, whose weighted score is the highest.
        var = ['variable','score']
        max_ = data[var].groupby(var[0]).agg(['max']).reset_index()
        
        # [1] Score.
        cond1 = data['variable'].isin(max_.values[:,0])
        cond2 = data['score'].isin(max_.values[:,1])
        data = data.loc[cond1 & cond2].reset_index(drop=True)
        
        # Select combination that comes first.
        var = ['variable','round']
        min_ = data[var].groupby(var[0]).agg(['min']).reset_index()
        
        # [2] If there is more than one candidate left, 
        # select the first one.
        cond1 = data['variable'].isin(min_.values[:,0])
        cond2 = data['round'].isin(min_.values[:,1])
        data = data.loc[cond1 & cond2].reset_index(drop=True)

        return data.drop(columns=['score']).to_dict(orient='list')

    def plot(self, column='round', values=None, adjusted=True, **plot_dict):

        '''
        Plot `WOE` (Weight of Evidence) from selected list. 
        Column selection can ONLY be made from either 
        'round' or 'variable'. Value must correpond to 
        selected columns (`str`, `int`, or list-like).

        Parameters
        ----------
        column : `str`, optional, default: 'round'
        \t It can only be either 'round' or 'variable'. If
        \t not in both categories, it defaults to `round`.
        
        values : `str` or `int` or list, optional, default: None
        \t If `int`, it indicates round number. If `str`, it 
        \t indicates variable name. `value` can be a list of 
        \t either `int` or `str`. If `None`, it takes all from
        \t log file.
    
        adjusted : `bool`, optional, default: True 
        \t If `True`, `self.data['best_bins']` is used  
        \t instead of `self.data['log']`.
        
        **plot_dict : keyword arguments
        \t Initial keyword arguments for `plot_woe` class.
        '''
        # Select list of bins to be displayed.   
        if adjusted: log = pd.DataFrame(self.data['best_bins'])
        else: log = pd.DataFrame(self.data['log'])
            
        # Get `data`.
        data = pd.DataFrame(self.data['data'])
        
        # Ensure that `column` points to the correct field.
        if column not in ['round','variable']: column = 'round'
            
        # Convert `int` or `str` input to `list`
        if values is None: values = np.unique(log[column])
        elif isinstance(values,(str, int)): values = [values]
        else: values = list(values)
        log_index = list(log.loc[log[column].isin(values)].index)

        if len(log_index) > 0:
            
            # `plot_woe` with arguments.
            woe = plot_woe(**{**_get_argument(plot_woe)[1],
                              **plot_dict})
            for i in log_index:
                log_ = log.loc[(log.index==i)]
                t = log_[['method','combination','round']].values.ravel()
                print('method : %s, combination : %d, round : %d' % tuple(t))
                woe.plot(data.loc[(data['round'].isin(log_['round']))])
        else:
            warn("Defined arguments result in an empty list." 
                 "Please make sure that either `column` or " 
                 "its corresponding `values` exists in "
                 "self.data['log']", UserWarning)

    def __to_df(self, X):

        '''
        [1] If `X` is an array with shape of (n_sample, n_features), 
            it will be transformed to dataframe. Name(s) will be 
            automatically assigned to all columns i.e. X1, X2, etc.
        [2] Remove variable(s) that contains only `np.nan` (missing) 
            and will be kept in `self.missing`.
        [3] Remove variable(s), apart from `np.nan` (missing),  
            contains only one value (like a constant) and will be kept 
            in `self.constant`.
            
        Parameters
        ----------
        X : `pd.DataFrame` object
        
        Returns
        -------
        `pd.DataFrame` with valid features.
        '''
        # Convert `X` to `pd.DataFrame`
        X = _to_DataFrame(X)
        columns = np.array(list(X))
        
        # Find percent missing and unique values for
        # respective variables.
        nan = (X.isna().sum(axis=0)/X.shape[0]==1)
        unq = np.array([len(X.loc[~X[var].isna(),var].unique())==1 
                        for var in X.columns])
        
        self.missing = columns[nan]
        self.constant = columns[unq]
        features = columns[(nan==False)&(unq==False)]
        
        if nan.sum()>0:
            warn("Variable(s) must not contain only `np.nan` "
                 "(missing). Got {} (see `self.missing`). "
                 .format(nan.sum()), UserWarning)
        
        if unq.sum()>0:
            warn("Variable(s) must not contain only single "
                 "value (constant). Got {} (see `self.constant`). "
                 .format(unq.sum()), UserWarning)
        
        return X[features]
    
    def _create_widgets(self):
        '''
        Create progress widgets.
        '''
        t1 = widgets.HTMLMath(value='Calculating...')
        t2 = widgets.HTMLMath(value='{:,.3g}%'.format(0))
        s_format = ' - ETA : {} - Method : {}'.format
        t3 = widgets.HTMLMath(value=s_format(np.nan,None))
        w = widgets.VBox([t1, widgets.HBox([t2,t3])])
        display(w)
        time.sleep(0.5)
        return t1, t2, t3, s_format

    def _sec_to_hms(self, seconds):
        '''
        Convert `seconds` to `H:M:S` format.
        '''
        h = seconds // 3600
        m = seconds % 3600 // 60
        s = seconds % 3600 % 60
        t = lambda n: '{:.0f}'.format(n).zfill(2)
        return '[{}h : {}m : {}s]'.format(t(h),t(m),t(s))

    def _eta(self, start, n, n_iters):
        '''
        Expected Time of Arrival.
        '''
        avg_time = (time.time()-start)/(n+1)
        eta = avg_time*abs(n_iters-n-1)
        return self._sec_to_hms(eta)

    def _progress_pct(self, n, n_iters):
        '''
        Percent progress in `str` format.
        '''
        p = min(int((n+1)/n_iters*100),100)
        return '{:,.3g}%'.format(p)
 
class woe_binning:
  
    '''
    ** Weight of Evidence **
    
    The `WOE` framework in credit risk is based on the following 
    relationship:

            log(P(y=0|X)/P(y=1|X)) <-- log-odds given X 
        =   log(P(y=0)/P(y=1))     <-- Sample log-odds
        +   log(f(X|y=0)/f(X|y=1)) <-- Weight of Evidence
    
    where f(X|y) denotes the conditional probability density 
    function or a discrete probability distribution if `X` is 
    categorical.
    
    In addition, when `WOE` is positive the chance of observing 
    `y`=1 is below average (for the sample), and vice versa when 
    `WOE` is negative. Basing on provided bins, `X` should have 
    a monotonic relationship to the dependent variable (`y`).
    
    In this class, there are 5 ways to achieve optimization of
    binning through defining `method` as follow;
    
    [1] `iv` : determine the cut-off with highest value of 
        information value.
        
    [2] `entropy` : use entropy to determine cut-off that returns
        the highest infomation gain.
        
    [3] `gini` : use gini-impurity to determine cut-off that has
        the least contaminated groups.
        
    [4] `chi` : chi-merge (supervised bottom-up) Using Chi-sqaure, 
        it tests the null hypothesis that two adjacent intervals 
        are independent. If the hypothesis is not accepted the 
        intervals are merged into a single interval, if not, they 
        remain separated.
        
    [5] `mono` : monotonic-optimal-binning. This method is fairly
        similar to `chi`. It uses Student's t-test (two independent 
        samples), it tests the null hypothesis that means of two 
        adjacent intervals are the same. If the hypothesis is not 
        accepted the intervals remain separated, if not, they are 
        merged into a single interval.
        
    .. versionadded:: 08-09-2020
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_breast_cancer as data
    
    >>> X, y = data(return_X_y=True)
    >>> columns = [s.replace(' ','_') for s in data().feature_names]
    >>> X = pd.DataFrame(X, columns=columns)

    # Use `woe_binning` to determine BIN.
    >>> model = woe_binning(method='gini', equal_bin=5)
    >>> model.fit(y, X['mean_texture'])
    >>> np.round(model_1.bin_edges, 0)
    array([10., 17., 18., 19., 20., 22., 40.])
    '''
    def __init__(self, trend='auto', n_order=1, random_state=None, 
                 equal_bin=None, n_step=20, sub_step=10, method='iv', 
                 min_pct=0.05, min_labels=(0.01,0.01),  
                 chi_alpha=0.05, p_value=0.05):
        '''
        Parameters
        ----------
        trend : `str`, optional, default: 'auto'
        \t This is relevant when either `iv`, `gini`, or `entropy`
        \t is selected for `method`. The followings are the values
        \t that must be specified;
        \t 'downward' : downward trend.
        \t 'upward' : upward trend. 
        \t 'auto' : allow function to determine trend automatically.
        
        n_order : `int`, optional, default: 1
        \t Order of selection. If `n_order` equals to `n`, it
        \t always select `cutoff` with the nth highest information 
        \t given `method` in each iteration. This only applies when 
        \t it is either `iv`, `gini`, or `entropy`. If `None`, 
        \t `n_order` is randomly generated based on availability of 
        \t options and random number (`random_state`).
        
        random_state : `int`, optional, default: None
        \t Determines random number generation for centroid 
        \t initialization. 

        equal_bin : `int`, optional, default: None
        \t Split `x` into equal-width bin. `equal_bin` must be
        \t `int` ranges from 1 to 6. For more info about method, 
        \t see print(woe_binning._equal_bin.__doc__). If `None`,
        \t `woe_binning._pct_bin` is applied instead.
        
        n_step : `int`, optional, default: 20 
        \t This is relevant when `equal_bin` is `None`. `n_step` 
        \t represents number of steps (`woe_binning._pct_bin`). 
        \t It is a starting interval (bins) of "Chi-merge", and
        \t "Monotonic Optimal Binning". 
        
        sub_step : `int`, optional, default: 10
        \t Number of steps for "Multi-Interval Discretization" 
        \t used for every iteration.
    
        method : `str`, optional, default: 'iv'
        \t Method of optimization.
        
        min_pct : `float`, optional, default: 0.05
        \t Minimum percentage of samples (event and non-event) in 
        \t each BIN. It defaults to 5%.
 
        min_labels : (`float`,`float`), optional, default: (0.01,0.01)
        \t Minimum positive percentage of event and non-event samples 
        \t with respect to their totals in each BIN.
        
        chi_alpha : `float`, optional, default: 0.05
        \t Significant level of Chi-sqaure (rejection region).
        \t It defaults to 5%.

        p_value : `float`, optional, default: 0.05
        \t Significant level of Student's T-Test (rejection region)
        \t It defaults to 5%.
        '''
        # ================================================
        # Generic parameters.
        self.min_pct = min_pct
        self.min_labels = min_labels
        self.method = method
        if isinstance(equal_bin,int):
            self.equal_bin = max(1,min(equal_bin,6))
        else: self.equal_bin = None
        # ------------------------------------------------
        # Multi-Interval Discretization (modified).
        self.trend, self.n_order = trend, n_order
        self.random_state, self.const = random_state, None
        self.sub_step = sub_step
        # ------------------------------------------------
        # Chi-Merge and Monotone Optimal Binning
        self.chi_alpha = chi_alpha
        self.p_value = p_value/2
        self.n_step = n_step
        # ------------------------------------------------
        # Validate parameters.
        self._check_values()
        # ================================================
   
    def fit(self, y, x):

        '''
        Fit model.
        
        (1) Multi-Interval Discretization (modified) 
            (method='iv','gini','entropy')
        (2) Chi-Merge (method='chi')
        (3) Monotone Optimal Binning (method='mono')
        
        Parameters
        ----------
        y : array or list of `int`
        \t Array of classes or labels.
        
        x : array or list of `float`
        \t Array of values to be binned.
        '''
        args = (np.array(y), np.array(x))
        if self.method in ['iv','gini','entropy']: 
            self._multi_inv_discretize(*args)
        elif self.method == 'chi' : 
            self._chi_merge(*args)
        elif self.method == 'mono': 
            self._monotone_optiomal_bin(*args)
  
    def _check_values(self):
        
        def is_num(num):
            return isinstance(num, numbers.Number)
        
        def check_num(param, name, limits):
            low, high = limits
            if not is_num(param) or not (low<param<high):
                raise ValueError("{} must be in the range ({}<x<{}). "
                                 "Got {}".format(name,low,high,param))
            
        def check_list(param, name, list_):
            if param not in list_:
                raise ValueError("{} can only be {}. Got ({}={})".
                                 format(name,list_,name,param))
                
        def random_int(high):
            random.seed(self.random_state)
            if self.const is None: 
                return random.randint(0,high-1)
            else: return min(self.const,high)-1
        
        # ========================================================
        # Check arguments from `list`.
        check_list(self.trend,'trend',['auto','downward','upward'])
        # --------------------------------------------------------
        check_list(self.method,'method', ['iv','gini','entropy',
                                          'chi','mono'])
        # --------------------------------------------------------
        # Check arguments from range.
        check_num(self.n_step,'n_step',[1,np.inf])
        check_num(self.sub_step,'sub_step',[1,np.inf])
        # --------------------------------------------------------
        if self.random_state is not None:
            check_num(self.random_state,'random_state',[0,np.inf])
        # --------------------------------------------------------
        if self.n_order is not None:
            check_num(self.n_order,'n_order',[0,np.inf])
            self.const = self.n_order
        self.n_order = random_int
        # --------------------------------------------------------
        check_num(self.min_pct,'min_pct',[0,1])
        # --------------------------------------------------------
        check_num(self.chi_alpha,'chi_alpha',[0,0.5])
        # --------------------------------------------------------
        check_num(self.p_value,'p_value',[0,1])
        # --------------------------------------------------------
        check_num(self.min_labels[0],'min_labels (0)',[0,1])
        check_num(self.min_labels[1],'min_labels (1)',[0,1])
        # ========================================================
                
    def _monotone_optiomal_bin(self, y, x):    

        '''
        ** Monotone Optimal Binning **
                
        [1] For every pair of adjacent bin p-value is 
            computed.
        [2] If number of samples in each class or number of 
            observations is less than defined limits add 1  
            to p-value. If a bin contains just one  
            observation then set p-value equals to 2.
        [3] Merge pair with highest p-value into single bin
        [4] Repeat (1), (2) and (3) until all p-values are 
            less than critical p-value.
        
        .. versionadded:: 08-09-2020
        
        References
        ----------
        [1] Pavel Mironchyk and Viktor Tchistiakov (2017), 
            Monotone optimal binning algorithm for credit 
            risk modeling, DOI: 10.13140/RG.2.2.31885.44003
        
        Parameters
        ----------
        y : array or list of `int`
        \t Array of classes or labels.
        
        X : array or list of `float`
        \t Array of values to be binned.
        
        Returns
        -------
        self.bin_edges
        '''
        # Initialize the overall trend and cut point.
        nonan_x, nonan_y = x[~np.isnan(x)], y[~np.isnan(x)]
        count = np.bincount(nonan_y)
    
        # Determine initial bin edges.
        bin_edges = self.__bin__(x, self.n_step, self.equal_bin)
        n_bins = len(bin_edges) + 1
  
        # ------------------------------------------------
        # Keep computing when bin_edges (t+1) is less than 
        # bin_edges (t) and number of bin_edges must be 
        # greater than 3 (2 bins).
        # ------------------------------------------------
        
        while (len(bin_edges)<n_bins) & (len(bin_edges)>3):
            
            n_bins = len(bin_edges)
            p_values = np.full(n_bins-2,0.0)
            
            for n in range(n_bins-1):
                
                if bin_edges[n:n+3].size==3:
                    # --------------------------------------------
                    # For each interation, the algorithm takes 2 
                    # bins (from both sides of cutoff) towards 
                    # comparison given samples criteria are 
                    # satisfied.
                    # --------------------------------------------
                    r_min, cutoff, r_max = bin_edges[n:n+3]
                    intvl = [(nonan_x>=r_min ) & (nonan_x<cutoff),
                             (nonan_x>=cutoff) & (nonan_x<r_max)]
                elif  bin_edges[n:n+3].size==2:
                    # --------------------------------------------
                    # Since the last bin does not have an adjacent
                    # bin to its right, thus, the one to left is 
                    # arbritarily used instead. The purpose is to
                    # avoid last bin having samples in each class
                    # below the defined thresholds.
                    # --------------------------------------------
                    n = n - 1
                    r_min, cutoff, r_max = bin_edges[n:n+3]
                    intvl = [(nonan_x>=cutoff) & (nonan_x<r_max),
                             (nonan_x>=r_min ) & (nonan_x<cutoff)]
                else:pass
                
                # Samples from both sides of cutoff.
                x1 = nonan_x[intvl[0]]
                x2 = nonan_x[intvl[1]]
                
                # (%) observations, events, and nonevents (current bin).
                pct_obs = nonan_x[intvl[0]].size/len(y)
                pct_cls = [nonan_y[intvl[0]&(nonan_y==n)].size/count[n] 
                           for n in [0,1]]
                
                # ======================================================
                # [1] If a bin contains just one observation then set 
                # ... p-value equals to 2.
                # [2] If number of samples in each class or number of  
                # ... observation is less than defined limits add 1 to 
                # ... p-value. 
                # [3] If none of above found, calculate p-value.
                cond1 = (x1.size<=1)
                cond2 = (pct_obs<self.min_pct) 
                cond2 = cond2 or (pct_cls[0]<self.min_labels[0])
                cond2 = cond2 or (pct_cls[1]<self.min_labels[1])
                # ------------------------------------------------------
                if cond1: p_values[n] = 2
                elif cond2: p_values[n] = 1
                elif (x2.size>1): 
                    p_values[n] = self.__independent_ttest(x1,x2)[1]
                # ======================================================

            # Merge pair with highest p-value into single bin.
            if max(p_values) > self.p_value:
                p_values = np.concatenate(([-np.inf],p_values,[-np.inf]))
                index = np.full(len(p_values),True)
                index[np.argmax(p_values)] = False
                bin_edges = bin_edges[index]

        # Check monotonicity.
        self.bin_edges = self._monotonic_woe(y, x, bin_edges)   
       
    def _chi_merge(self, y, x):

        '''
        ** Chi-merge **
        
        (1) For every pair of adjacent bin a χ2 values is 
            computed.
        (2) Merge pair with lowest χ2 (highest p-value) 
            into single bin.
        (3) Repeat (1) and (2) until all X2 are more than 
            predefined threshold. The minimum number of 
            bins is 2.
        
        .. versionadded:: 08-09-2020
        
        References
        ----------
        [1] Pavel Mironchyk and Viktor Tchistiakov (2017), 
            Monotone optimal binning algorithm for credit 
            risk modeling, DOI: 10.13140/RG.2.2.31885.44003
            
        Parameters
        ----------
        y : array or list of `int`
        \t Array of classes or labels.
        
        X : array or list of `float`
        \t Array of values to be binned.
        
        Returns
        -------
        self.bin_edges
        '''
        # Exclude all missings.
        nonan_x = x[~np.isnan(x)] 
        nonan_y = y[~np.isnan(x)]
        count = np.bincount(nonan_y)
        
        # Degrees of freedom.
        dof = len(np.unique(y)) - 1
        
        # Rejection area (p-value).
        threshold = chi2.isf(self.chi_alpha, df=dof)
        p_value = 1-chi2.cdf(threshold, df=dof) 
        
        # Determine initial bin edges.
        bin_edges = self.__bin__(x, self.n_step, self.equal_bin)
        n_bins = len(bin_edges) + 1
        
        # ------------------------------------------------
        # Keep computing when bin_edges (t+1) is less than 
        # bin_edges (t) and number of bin_edges must be 
        # greater than 3 (2 bins).
        # ------------------------------------------------
        
        while (len(bin_edges)<n_bins) & (len(bin_edges)>3):
            
            n_bins = len(bin_edges)
            crit_val = np.full(n_bins-2,0.0)
            
            # For every pair of adjacent bin a χ2 values is computed.
            for n in range(n_bins-1):
                
                if bin_edges[n:n+3].size==3:
                    # --------------------------------------------
                    # For each interation, the algorithm takes 2 
                    # bins (from both sides of cutoff) towards 
                    # comparison given samples criteria are 
                    # satisfied.
                    # --------------------------------------------
                    r_min, cutoff, r_max = bin_edges[n:n+3]
                    intvl = [(nonan_x>=r_min ) & (nonan_x<cutoff),
                             (nonan_x>=cutoff) & (nonan_x<r_max)]
                elif  bin_edges[n:n+3].size==2:
                    # --------------------------------------------
                    # Since the last bin does not have an adjacent
                    # bin to its right, thus, the one to left is 
                    # arbritarily used instead. The purpose is to
                    # avoid last bin having samples in each class
                    # below the defined thresholds.
                    # --------------------------------------------
                    n = n - 1
                    r_min, cutoff, r_max = bin_edges[n:n+3]
                    intvl = [(nonan_x>=cutoff) & (nonan_x<r_max),
                             (nonan_x>=r_min ) & (nonan_x<cutoff)]
                else:pass
                
                # Samples from both sides of cutoff.
                x1 = nonan_x[intvl[0]]
                x2 = nonan_x[intvl[1]]
                args = (nonan_y, nonan_x, r_min, r_max, cutoff)
                
                # (%) observations, events, and nonevents (current bin).
                pct_obs = nonan_x[intvl[0]].size/len(y)
                pct_cls = [nonan_y[intvl[0]&(nonan_y==n)].size/count[n] 
                           for n in [0,1]]
                
                # ======================================================
                # [1] If a bin contains just one observation then set 
                # ... critical-value equals to -2.
                # [2] If number of samples in each class or number of  
                # ... observation is less than defined limits add -1 to 
                # ... critical-value. 
                # [3] If none of above found, calculate critical-value.
                cond1 = (x1.size<=1)
                cond2 = (pct_obs<self.min_pct) 
                cond2 = cond2 or (pct_cls[0]<self.min_labels[0])
                cond2 = cond2 or (pct_cls[1]<self.min_labels[1])
                # ------------------------------------------------------
                if cond1: crit_val[n] = -2
                elif cond2: crit_val[n] = -1
                else: crit_val[n] = self.__chi_square(*args)[0]
                # ======================================================
            
            # Merge pair with lowest χ2 (highest p-value) into single bin.
            if min(crit_val) < threshold:
                crit_val = np.concatenate(([np.inf],crit_val,[np.inf]))
                index = np.full(len(crit_val),True)
                index[np.argmin(crit_val)] = False
                bin_edges = bin_edges[index]
        
        # Check monotonicity.
        self.bin_edges = self._monotonic_woe(y, x, bin_edges)    
        
    def _multi_inv_discretize(self, y, x):
    
        '''
        ** Multi-Interval Discretization (modified) **
        
        List of cutoffs is determined in this instance 
        basing on 3 different indicators, which are 
        `Information Value`, `Entropy`, and `Gini Impurity`. 
    
        (1) For each interval, divide input into several 
            cut points. 
        (2) for each cut point, compute the indicator that 
            satisfies trend of WOEs and optimizes objective 
            function. If `None`, no cutoff is selected for
            this bin.
        (3) Repeat (1) and (2) until no cutoff is made. The
            minimum number of bins is 2.
            
        .. versionadded:: 08-09-2020
        
        Parameters
        ----------
        y : array or list of `int`
        \t Array of classes or labels.
        
        x : array or list of `float`
        \t Array of values to be binned.
        
        Returns
        -------
        self.bin_edges
        '''
        # Initialize the overall trend and cut point.
        r_min , r_max = np.nanmin(x), np.nanmax(x) * 1.01
        cutoff, _ = self.__find_cutoff(y, x, r_min, r_max)
        bin_edges = np.unique([r_min, cutoff, r_max]).tolist()
        n_bins = 0
        
        if len(bin_edges) > 2:
            
            while len(bin_edges) > n_bins:
                n_bins = len(bin_edges)
                new_bin_edges = list()
                
                for n in range(n_bins-1):
                    r_min, r_max = bin_edges[n:n+2]
                    cutoff, _ = self.__find_cutoff(y, x, r_min, r_max)
                    
                    # Store cutoff when it is not equal to `r_min`.
                    if cutoff != r_min: new_bin_edges.append(cutoff) 
                
                # Store `new_bin_edges`.
                bin_edges.extend(new_bin_edges)
                bin_edges = np.unique(np.sort(bin_edges,axis=None)).tolist()  
                
        else: bin_edges = [r_min, np.median(x[~np.isnan(x)]), r_max]
            
        # Check monotonicity.   
        self.bin_edges = self._monotonic_woe(y, x, bin_edges) 
 
    def __find_cutoff(self, y, x, r_min, r_max):

        '''
        Finds the optimum cut point that satisfies the 
        objective function. (optimize defined indicator 
        e.g. IV).
        
        Parameters
        ----------
        y : array or list of `int`
        \t Array of classes or labels.
        
        x : array or list of `float`
        \t Array of values to be binned.
      
        r_min : `float`
        \t `r_min` is a minimum value of X (r_min ≤ X).
        
        r_max : `float`
        \t `r_max` is a maximum value of X (X > r_max).
        
        Returns
        -------
        - cutoff : `float`
        - woe_corr : `float`
        '''  
        # Default values.
        cutoff, woe_corr = r_min, -1
        direction = {'downward':0.0,'upward':1.0}
        
        # Determine WOEs between bins.
        woe = self._woe_btw_bins(y, x, r_min, r_max)
        
        # If self._woe_btw_bins returns `None`.
        if woe.shape[0]>0:
            
            # If trend is not defined i.e. "downward" or "upward",
            # it will determine by counting `corr_index` and take
            # direction that has a majority of counts.
            if self.trend=='auto': 
                woe_corr = (woe['corr_index']==1).sum()/woe.shape[0]
            else: woe_corr = direction[self.trend]

            # select entry that corresponds to woe_corr
            if woe_corr >= 0.5: 
                cond = (woe['corr_index']==1)
                self.trend = 'upward'
            elif woe_corr < 0.5: 
                cond = (woe['corr_index']==0)
                self.trend = 'downward'
            
            # Select records that follow the trend.
            woe = woe.loc[cond, ['value','cutoffs']]
            woe = woe.sort_values(['value'], ascending=False).values[:,1]
            if len(woe) != 0: cutoff = woe[self.n_order(len(woe))]
                
        return cutoff, woe_corr

    def _woe_btw_bins(self, y, x, r_min, r_max):

        '''
        Determines list of WOEs (Weight of Evidence) from 
        different cut points. This is applicable only to 
        Multi-Interval Discretization i.e. `iv`, `entropy` 
        and `gini`.
        
        .. versionadded:: 08-09-2020
        
        Parameters
        ----------
        y : array or list of `int`
        \t Array of classes or labels.
        
        x : array or list of `float`
        \t Array of values to be binned.
      
        r_min : `float`
        \t `r_min` is a minimum value of X (r_min ≤ X).
        
        r_max : `float`
        \t `r_max` is a maximum value of X (X > r_max).
        
        Returns
        -------
        self.woe_df : `pd.DataFrame` object
        '''
        # Necessary functions.
        def WOE(a): return np.log(a[0]/a[1]) if a[1]>0 else 0
        def IV(a): return (a[0]-a[1])*WOE(a)

        if  self.method in ['iv','gini','entropy']:
            
            # Find total number of samples for classes.
            n_event =  np.bincount(y) 
            
            # Accept values that are not null and stay within 
            # defined boundary.
            cond = ((~np.isnan(x)) & (r_min<=x) & (x<r_max))
            y = np.array(y).ravel()[cond]
            x = np.array(x).ravel()[cond]
            
            # Determine list of cutoffs given number of bins.
            bin_cutoff = self.__bin__(x, self.sub_step, self.equal_bin)
            
            # Prepare list of outcomes from all valid cutoffs.
            left_woe, right_woe, new_bin, gain = [],[],[],[]
            
            # Number of bin_cutoff (r_min and r_max excluded).
            if len(bin_cutoff)-2 > 0:
                for cutoff in bin_cutoff[1:-1]:
                    
                    # Distributions of event and non-event
                    arg = (y, x, cutoff, n_event)
                    left, right, a, b = self._two_interval_dist(*arg)
                    
                    # =======================================================
                    # In order for WOE to be valid, 2 following 
                    # conditions must be satisfied;
                    # [1] Bin should contain at least 5% observations
                    # ... or be greater than or equal to self.min_pct.
                    # [2] Bin must not have 0 accounts for all classes
                    # ... or be greater than or equal to self.min_labels.
                    # -------------------------------------------------------
                    cond1 = (self.min_pct <= min(a,b))
                    cond2 = (self.min_labels[0] <= min(left[0], right[0]))
                    cond3 = (self.min_labels[1] <= min(left[1], right[1]))
                    conditions = cond1 & cond2 & cond3
                    # =======================================================
                    
                    if conditions:
                        
                        left_woe.append(WOE(left))
                        right_woe.append(WOE(right))
                        new_bin.append(cutoff)
                        
                        # ==============================================
                        if self.method == 'iv': 
                            gain.append(IV(left) + IV(right))
                        # ----------------------------------------------
                        elif self.method == 'entropy':
                            begin = self._entropy(y, x)[1]
                            split = self._entropy(y, x, cutoff)[1]
                            gain.append(begin-split)
                        # ----------------------------------------------
                        elif self.method == 'gini':
                            begin = self._gini(y, x)[1]
                            split = self._gini(y, x, cutoff)[1]
                            gain.append(begin-split)
                        else: pass
                        # ==============================================
                        
            else: pass
            
            # Convert results to `pd.DataFrame`.
            a = pd.DataFrame({'left_woe':left_woe, 'right_woe':right_woe, 
                              'value':gain, 'cutoffs':new_bin})
            # woe_corr : {1: positive, 0: negative}
            a['corr_index'] = np.where((a.left_woe<=a.right_woe),1,0)
            self.woe_df = a.copy();del a
            return self.woe_df
        else:
            warn("_woe_btw_bins can be used when method is 'iv','gini', "
                 "and 'entropy'. Got (method={})".format(self.method))
            return None

    def _two_interval_dist(self, y, x, cutoff, n_event):

        ''' 
        Determines distribution of events and non-events 
        from 2 intervals given cut point.
        
        Parameters
        ----------
        y : array or list of `int`
        \t Array of classes or labels.
        
        X : array or list of `float`
        \t Input array.
        
        cutoff : `float`
        \t left-side ≤ `cutoff`, right-side > `cutoff`.
        
        n_event : list of `int`, [`int`, `int`]
        \t `n_event` is number of samples in each class 
        \t from entire sample.
        
        Returns
        -------
        - List of left-bin (%) by class: [`float`, `float`]
        - List of right-bin (%) by class: [`float`, `float`]
        - % of samples in left bin : `float`
        - % of samples in right bin : `float`
        '''   
        left = [sum((x<cutoff)&(y==n))/
                float(n_event[n]) for n in [0,1]]
        right= [sum((cutoff<=x)&(y==n))/
                float(n_event[n]) for n in [0,1]]
        nl_bin = sum(x<cutoff)/sum(n_event) # <-- % left dist.
        nr_bin = sum(cutoff<=x)/sum(n_event) # <-- % right dist.
        return left, right, nl_bin, nr_bin

    def __bin__(self, x, bins=20, method=3):
        
        if isinstance(method,int): 
            return self._equal_bin(x, method)
        else: return self._pct_bin(x, bins)
    
    def _pct_bin(self, x, bins=20):

        ''' 
        ** Creates percentile Bins **
        
        [1] Each bin should contain at least 5% of observations
        [2] In case when X contains the same value more than 90% 
            or more, this affects percentile binning to provide 
            an unreasonable range of data. Thus, it will switch 
            to equal-binning method to group those with the same 
            value into one bin while the others will spread out.
        
        .. versionadded:: 08-09-2020
        
        Parameters
        ----------
        x : array or list of `float`
        \t Array of values to be binned.
        
        bins : `int`, optional, default: 20
        \t Number of bins.
        
        Returns
        -------
        bin_edges : array of float
        \t The bin edges (length(hist)+1).
        '''
        a, b = x[~np.isnan(x)], 100/float(bins)
        q = [min(n*b,100) for n in range(bins+1)]
        bin_edges = np.unique(np.percentile(a,q))
        bin_edges[-1:] = bin_edges[-1:] + 1
        if len(bin_edges)<=2:
            return self._equal_bin(a,3)
        return bin_edges
  
    def _equal_bin(self, x, method=3):

        '''
        Determine the equal bins.
        [1] `Square-root choice`: takes the square root of the 
            number of data points in the sample.
        [2] `Sturge's formula` is derived from a binomial  
            distribution and implicitly assumes an  approximately 
            normal distribution.
        [3] `The Rice Rule` is presented as a simple  alternative 
            to `Sturges's rule`.
        [4] `Doane's formula` is a modification of `Sturges' 
            formula`, which attempts to improve  its performance 
            with non-normal data.
        [5] `Scott's normal reference rule`.
        [6] `Freedman–Diaconis' choice`.
        
        .. versionadded:: 08-09-2020
        
        Parameters
        ----------
        x : array or list of `float`
        \t Array of values to be binned.
        
        method : `int`, optional, default: 3
        \t 1 : `Square-root choice`,
        \t 2 : `Sturge's formula`,
        \t 3 : `The Rice Rule`,
        \t 4 : `Doane's formula`,
        \t 5 : `Scott's normal reference rule`,
        \t 6 : `Freedman–Diaconis' choice`
      
        Returns
        -------
        bin_edges : array of float
        \t The bin edges (length(hist)+1). 
        '''
        a = x[~np.isnan(x)].copy()
        v_max, v_min, v_sum = max(a), min(a), sum(a)
        sigma, N = np.std(a), len(a)
        method = max(min(method,6),1)
        
        # ===================================================
        if method==1 : bins = np.sqrt(N)
        # ---------------------------------------------------
        elif method==2 : bins = np.ceil(np.log2(N)+1)
        # ---------------------------------------------------
        elif method==3 : bins = 2*(N**(1/3))
        # ---------------------------------------------------
        elif method==4 :
            s = abs(((sum(a)/N)-np.median(a))/sigma)
            s = s/np.sqrt(6*(N-2)/((N+1)*(N+3)))
            bins = 1 + np.log2(N) + np.log2((1+s))
        # ---------------------------------------------------
        elif method==5 :
            bin_width = 3.5*sigma/(N**(1/3))
            if bin_width > 0: bins = (v_max-v_min)/bin_width
            else: bins = np.ceil(np.log2(N)+1)
        # ---------------------------------------------------
        elif method==6 :
            iqr = np.diff(np.percentile(a,[25,75]))
            bin_width = 2*iqr/(N**(1/3))
            if bin_width > 0: bins = (v_max-v_min)/bin_width
            else: bins = np.ceil(np.log2(N)+1)
        else: pass
        # ===================================================
        
        # Round up number of bins.
        bins = max(int(np.ceil(bins)),2)
        bin_width = (v_max-v_min)/bins
        bin_edges = [min(v_min+(n*bin_width),v_max) 
                     for n in range(bins+1)]
        bin_edges[-1] = bin_edges[-1] + 1
        
        return np.unique(bin_edges)

    def _entropy(self, y, x, cutoff=None):

        '''
        Multi-interval Discretization (entropy).
        
        .. versionadded:: 08-09-2020
        
        Parameters
        ----------
        y : array or list of `int`
        \t Array of classes or labels.
        
        x : array or list of `float`
        \t Array of values to be binned.
       
        cutoff : float, optional, default: None
        \t i.e. left-side ≤ `cutoff`, and right-side > `cutoff`.
        \t If `None`, `cutoff` is default to `np.max`.
        
        Returns
        -------
        - Array of left and right weighted Entropies.
        - Sum of weighted Entropies : `float`
        '''
        if cutoff==None: cutoff=max(x)
        n_class, n_cnt = np.unique(y), float(len(y))
        y, x = np.array(y).ravel(), np.array(x).ravel()
        cond, entropy_ = [(x<=cutoff),(x>cutoff)], np.zeros(2)
        
        for n in range(2):
            n_cutoff = float(max(cond[n].sum(),1))    
            p_cutoff, entropy = n_cutoff/n_cnt, 1
            entropy = [((cond[n]&(y==c)).sum()/n_cutoff)**2 
                       for c in n_class]
            entropy = [e*np.log2(e) if e>0 else 0 for e in entropy]   
            entropy_[n] = p_cutoff * (1-sum(entropy))
        return entropy_, sum(entropy_)

    def _gini(self, y , x, cutoff=None):

        '''
        Multi-interval Discretization (gini-impurity).
        
        .. versionadded:: 08-09-2020
        
        Parameters
        ----------
        y : array or list of `int`
        \t Array of classes or labels.
        
        x : array or list of `float`
        \t Array of values to be binned.
       
        cutoff : float, optional, default: None
        \t i.e. left-side ≤ `cutoff`, right-side > `cutoff`.
        \t If `None`, `cutoff` is default to `np.max`.
        
        Returns
        -------
        - Array of left and right weighted GINIs.
        - Sum of weighted GINIs : `float`
        '''
        if cutoff==None: cutoff=max(x)
        n_class, n_cnt = np.unique(y), float(len(y))
        y, x = np.array(y).ravel(), np.array(x).ravel()
        cond, gini_ = [(x <= cutoff),(x > cutoff)], np.zeros(2)
        
        for n in range(2):
            n_cutoff = float(max(cond[n].sum(),1))    
            p_cutoff, gini = n_cutoff/n_cnt, 1   
            gini = [((cond[n]&(y==c)).sum()/n_cutoff)**2 for c in n_class]
            gini_[n] = p_cutoff * (1-sum(gini))
        return gini_, sum(gini_)

    def __chi_square(self, y, x, r_min=None, r_max=None, cutoff=None):

        '''
        Chi-Square (χ2) is used to test whether sample data 
        fits a distribution from a certain population or 
        not. Its null hypothesis or `H0` says that the 
        observed population fits the expected population.
        
        .. versionadded:: 08-09-2020
        
        Parameters
        ----------
        y : array of `int`
        \t Array of classes or labels.
        
        x : array of `float`
        \t Array of `float` to be binned.
        
        r_min : `float`, optional, (default:None)
        \t `r_min` is a minimum value of `x` (r_min ≤ `x`).
        \t If `None`, `np.nanmin` of `x` is assigned.
        
        r_max : `float`, optional, (default:None)
        \t `r_max` is a maximum value of X (X > r_max).
        \t If `None`, `np.nanmax` of `x` is assigned.
        
        cutoff : `float`, optional, (default:None)
        \t i.e. left-side ≤ `cutoff`, right-side > `cutoff`.
        \t If `None`, (r_max - r_min)/2 is assigned.
        
        Returns
        -------
        cv : `float`
        \t `cv` is a critival value. If the critical value
        \t from the table given `degrees of freedom` and `α` 
        \t (rejection region) is less than the computed 
        \t critical value (`cv`), then the observed data does
        \t not fit the expected population or in another word, 
        \t we reject to accept the null hypothesis.

        p : `float`
        \t `p` is a p-value that corresponds to `cv`.
        '''
        if r_min is None: r_min = np.nanmin(x)
        if r_max is None: r_max = np.nanmax(x)
        if cutoff is None: cutoff = (r_max-r_min)/2
        
        # Get x, and y.
        cond = (r_min<=x) & (x<r_max)
        y = np.array(y)[cond].ravel()
        x = np.array(x)[cond].ravel()
        
        n_class, n_cnt = np.unique(y), float(len(y))
        conds, crit_val = [(x<=cutoff),(x>cutoff)], 0
        n_R, n_C = [conds[n].sum() for n in range(2)], np.bincount(y)
        
        for n,a in enumerate(conds):
            for m,b in enumerate(n_class):
                C_ab = len(x[a & (y==b)])
                E_ab = float(n_R[n]*n_C[m]/n_cnt)
                if E_ab > 0: crit_val += (C_ab-E_ab)**2/E_ab

        return crit_val, 1 - chi2.cdf(crit_val, df=len(n_class)-1)

    def __independent_ttest(self, x1, x2):

        '''
        Assuming unequal variances, Two-sample t-test, whose
        null hypothesis or `H0` is μ1 = μ2 and alternative 
        hypothesis `HA` is μ1 ≠ μ2.

        .. versionadded:: 26-08-2020

        Parameters
        ----------
        x1, x2 : array-like (1-dimensional) of `float`
        \t Input data that assumed to be drawn from a  
        \t continuous distribution. Sample sizes can be 
        \t different.

        Returns
        -------
        t_stat : `float` 
        \t `t-statistic` is used to determine whether to 
        \t accept or reject to accept the null hypothesis.

        p : `float`
        \t `p` is a one-tailed p-value that corresponds to 
        \t `t_stat`. We accept the null hypothesis (μ1 = μ2) 
        \t when `p` ≤ α/2 (rejection region), otherwise we 
        \t reject to accept the null hypothesis (μ1 ≠ μ2).
        '''
        # (1) Standard errors.
        se = lambda x : np.nanstd(x,ddof=1)/np.sqrt(len(x))
        
        # (2) Standard errors of two distributions.
        sed = lambda x1,x2 : np.sqrt(se(x1)**2 + se(x2)**2)
        
        # (3) When `x1` and `x2` are constant, `sed` is 0.
        #     This also results t-statistic to be 0.
        mu = lambda x : np.nanmean(x)
        tstat = lambda x1,x2,s : (mu(x1)-mu(x2))/s if s>0 else 0

        # Calculate t-statistic.
        t_stat = tstat(x1,x2,sed(x1,x2)) 
        
        # Calculate degree of freedom.
        a = np.array([se(x1)**2/len(x1), se(x2)**2/len(x2)])
        b = np.array([1/(len(x1)-1), 1/(len(x2)-1)])
        if a.sum()>0: df = np.floor(a.sum()/(a*b).sum())
        else: df = len(x1) + len(x2) - 2
            
        return abs(t_stat), 1-t.cdf(abs(t_stat), df)  
    
    def _monotonic_woe(self, y, x, bins):
        
        '''
        In case that `WOE` don't form a monotonic trend either
        upward or downward (majority vote), `self._monotonic_woe` 
        will collaspe `BIN` (to the left) that does not follow  
        trend one at a time, until either all `WOE` are in the 
        same direction or number of bins equals to 2, whatever 
        comes first. 
        
        .. versionadded:: 08-09-2020
        
        Parameters
        ----------
        y : array of `int`
        \t Array of classes or labels.
        
        X : array of `float`
        \t Array of `float` to be binned.
        
        bins : list or array of `float`
        \t Bin edges.
        
        Returns
        -------
        Monotonic bin edges : list of `float`.
        '''
        # `np.nan` must not be considered as part
        # of monotomicity, thus it will be removed.
        y = np.array(y)[~np.isnan(x)]
        x = np.array(x)[~np.isnan(x)]
    
        # Loop until bins remain unchanged or number
        # of bins is less than 3.
        bins, n_bins = np.array(bins), 0
        cnt = np.bincount(y)

        while (len(bins)!=n_bins) and len(bins)>3:
            
            # Calculate `WOE`.
            n_bins = len(bins)
            hist = [np.histogram(x[y==c],bins)[0] 
                    for c in range(2)]
            hist = [np.where(h==0,0.5,h)/cnt[n] 
                    for n,h in enumerate(hist)]
            woes = np.log(hist[0]/hist[1])
            
            # Determine trend of `WOE`.
            direction = np.sign(np.diff(woes))
            
            # There is an adjacent bin, whose `WOE` equals.
            # The current bin will be collasped to the left.
            if (direction==0).sum()>0: 
                index = np.full(len(direction),True)
                index[np.argmax((direction==0))] = False
            else:   
                index = np.where(direction<0,0,1) 
                index = (index==np.argmax(np.bincount(index)))
                
            # Keep bin edges that follow the trend.
            bins = bins[np.hstack(([True],index,[True]))]
         
        return bins
    
class evaluate_bins:
    
    '''
    ** evaluate_bins **
    
    This class evaluates the predictiveness of bin intervals 
    by using Weight-of-Evidence (WOE), Information Value (IV),
    Spearman rank-order Correlation, and intercept from 
    regression model (compared against log(event/non-event)).
    
    .. versionadded:: 08-09-2020
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_breast_cancer as data
    
    >>> X, y = data(return_X_y=True)
    >>> X = pd.DataFrame(X, columns=[s.replace(' ','_') 
    ...                              for s in data().feature_names])

    # Use `woe_binning` to determine BIN.
    >>> binning = woe_binning(method='gini')
    >>> binning.fit(y, X['mean_texture'])

    # Use `evaluate_bins` to construct `woe_df`.
    >>> eval_bin = evaluate_bins()
    >>> eval_bin.fit(y, X['mean_texture'], binning.bin_edges)
    
    # Results: WOE dataframe.
    >>> eval_bin.woe_df
    
    # Information Value
    >>> eval_bin.iv
    
    # Intercepts from Logistic Regression.
    >>> eval_bin.intercept_
    '''
    def __init__(self):
        
        '''
        ** No initial inputs required **
        '''
        pass
    
    def fit(self, y, x, **params):
        
        '''
        Fit model.
        
        Parameters
        ----------
        y : array of shape (n_samples,)
        \t Target values (binary).
        
        x : array or `pd.Series`, of shape (n_samples,)
        \t Raw data that will be binned according to bin edges.
        
        **params : dictionary of properties, optional
        \t params are used to specify or override properties of 
    
            bins : `int` or `list` of `float`, optional, default: 10
            \t If `int`, it defines the number of equal-width bins 
            \t in the given range. If bins is a sequence, it defines
            \t a monotonically increasing array of bin edges,  
            \t including the rightmost edge.

            replace_null : float, optional, default: 0.5
            \t For any given range, if number of samples for either 
            \t classes equals to `0` (0%), it will be replaced by 
            \t `replace_null`. This is meant to avoid error when  
            \t `WOE` is calculated.

        Returns 
        -------
        self.woe_df : `pd.Dataframe` object
        \t `WOE` dataframe.
        
        self.iv : `float`
        \t Information Value (IV).
        
        self.rho : `float`
        \t Spearman rank-order correlation. Exluding missing, it 
        \t measures the correlation between `X` against 
        \t Weight-of-Evidence. If `WOE` ranks monotonically, 
        \t correlation must be high (>0.5).
        
        self.intercept_ : (`float`,`float`,`float`) 
        \t Tuple of floats, which are arranged as following   
        \t (1) model intercept obtained from fitting `y` and `X`.
        \t     This is implementing `LogisticRegression` from 
        \t     `sklearn.linear_model`.
        \t (2) log of target over non-target (bias).
        \t (3) absolute difference between intercepts.  
        '''
        # Weight of Evidence.
        self.woe_df = _woe_table_(y, x, **params)
        woe = _assign_woe(x, self.woe_df)
        
        # Information Value
        self.iv = self.woe_df['iv'].sum()
        
        # Spearman rank-order 
        self.rho = _check_correl_(self.woe_df)

        # Check binning soundness (intercpet)
        self.intercept_ = _check_binning_(y, woe)

def _woe_table_(y, x, bins=10, replace_null=0.5):
        
    '''
    Construct `WOE` dataframe.

    Parameters
    ----------
    y : array-like or `pd.Series`
    \t Target values (binary).

    x : `pd.Series`
    \t Input array of `float` to be binned.
    
    bins : `int` or `list` of `float`, optional, default: 10
    \t If `int`, it defines the number of equal-width bins 
    \t in the given range. If bins is a sequence, it defines a 
    \t monotonically increasing array of bin edges, including 
    \t the rightmost edge.

    replace_null : float, optional, default: 0.5
    \t For any given range, if number of samples for either 
    \t classes equals to `0` (0%), it will be replaced by 
    \t `replace_null`. This is meant to avoid error when `WOE` 
    \t is calculated.

    Returns
    -------
    woe_df : `pd.DataFrame` object.
    '''
    if not isinstance(x, pd.Series):
        raise ValueError("'x' must be `pd.Series`. "
                         "Got {}".format(type(x)))
    
    # If `bins` is `int`, determine equal-width bins.
    if isinstance(bins,int): 
        bins = np.histogram(x[~np.isnan(x)],bins=bins)[1]
        bins[0] = -np.inf; bins[-1] = np.inf
    elif not isinstance(bins,(list, np.ndarray)):
        raise ValueError("`bins` must be array-like."
                         " Got {}".format(type(bins)))
    else: bins = np.array(bins)

    # `np.digitize()` assigns len(bins) to `np.nan`.
    # Since missing values in `woe_df` is in bin `0`,
    # such value has to be change accordingly.
    a = np.digitize(x, bins, right=False)
    a[a==len(bins)] = 0 
    
    # Find number of samples in in each BIN. 
    # Not every bin will have sample.
    k = dict(return_counts=True)
    counts = [np.unique(a[y==c], **k) for c in [0,1]]

    # Allocate number of samples to BIN.
    cnt, pct = [None]*2, [None]*2
    for (n,a) in enumerate(counts):
        cnt[n] = np.full(len(bins),0)
        cnt[n][np.isin(np.arange(len(bins)),a[0])] = a[1]
        # Percent distribution (%).
        pct[n] = cnt[n]/sum(cnt[n]) 

    # Replace 0 with `replace_null` by class [0,1].
    null = replace_null
    dist = [np.where(n==0, null/sum(n) , n/sum(n)) for n in cnt]
    
    # If BIN has no sample, `WOE` must be 0.
    woe = np.where(sum(pct)==0,0,np.log(dist[0]/dist[1]))

    # Calculate `Information Value`.
    iv = (pct[0]-pct[1])*woe
    
    # BIN Intervals [min, max].
    min_ = [np.nan, -np.inf] + bins[1:-1].tolist()
    max_ = [np.nan] + bins[1:-1].tolist() + [np.inf]
    n_bin = np.arange(len(bins))

    # `WOE` dataframe.
    data = {'variable':np.full(len(cnt[0]),x.name), 
            'min':min_, 'max':max_, 'bin':n_bin, 
            'non_events':cnt[0], 'events':cnt[1], 
            'pct_nonevents':pct[0], 'pct_events':pct[1], 
            'woe':woe, 'iv':iv}
    
    return pd.DataFrame(data)    
      
def _check_binning_(y, x, **params):

    '''
    A necessary condition for a good binning is that 
    β0 = log(Target/Non-Target) or β1 (slope) = 1 when a
    logistic regression model is fitted with one independent 
    variable that has undergone a WOE transformation. If 
    aforementioned conditions are not satisfied, then it could 
    be that binning algorithm is not working properly.
    
    References
    ----------
    [1] http://www.m-hikari.com/ams/ams-2014/ams-65-68-2014/
        zengAMS65-68-2014.pdf, Applied Mathematical Sciences, 
        Vol. 8, 2014, no. 65, 3229 - 3242, Guoping Zeng.
    
    Parameters
    ----------
    y : array-like or `pd.Series`
    \t Target values (binary).

    x : array-like or `pd.Series`
    \t Input array of `WOE`.
    
    **params : dictionary of properties, optional
    \t params are used to specify or override properties of 
    
        estimator : `estimator` object
        \t Logistic regression estimator i.e. 
        \t `sklearn.linear_model.LogisticRegression`.

    Returns
    -------
    `dictionary` object with following keys;
    
    `intercept` : `float`
    \t Intercept (β0) from `LogisticRegression`.
    
    `bias` : `float`
    \t `bias` computed from np.log(Target/Non-Target).
    
    `delta` : `float`
    \t Percent difference (%) between `intercept` and `bias`.
    \t i.e. |(intercept - bias) / bias|
    
    beta : `float`
    \t |β1| (slope) from logistic regression.

    Notes
    -----
    Due to regularization `l1`, this logistic regression 
    will never return coefficient as normal logistic does.
    '''
    # Calculate sample bias.
    bins = np.bincount(y)
    log_intercept = np.log(bins[1]/bins[0])
    
    # Calculate bias from logistic regression.
    intercept_ = np.nan; coef_ = np.nan
    try:
        # Determine `estimator` from `params`.
        logit = params.get('estimator')
        if logit is None:
            logit = LogisticRegression(solver='liblinear', 
                                       fit_intercept=True, 
                                       penalty='l1')
        logit.fit(x.reshape(-1,1), y)
        intercept_ = float(logit.intercept_)
        coef_ = abs(float(logit.coef_[0]))
    except: pass
        
    # Percent difference (%) between `Intercept` and `bias`.  
    diff = abs((intercept_-log_intercept)/log_intercept)
    
    return {'intercept':intercept_, 'bias':log_intercept, 
            'delta':diff, 'beta':coef_}

def _check_correl_(woe_df):

    '''
    This method measures the correlation between `WOE` 
    and target rate. It takes the minimum of absolute
    values from three different methods i.e. Pearson, 
    Kendall’s Tau, and Spearman's. 

    Parameters
    ----------
    woe_df : `pd.Dataframe` object.
    \t `pd.DataFrame` with mandatory fields i.e. 'min', 
    \t 'max', 'bin', and 'woe'. `woe_df` is one of the 
    \t attributes from `evaluate_bins.woe_df`.

    Returns
    -------
    Spearman rank-order correlation : `float`
    '''
    x1 = woe_df['woe'].values[1:]
    sample = woe_df[['non_events','events']].sum(axis=1)
    x2 = woe_df['events'].values[1:]/sample.values[1:]
    
    try:
        rho = [pearsonr(x1, x2)[0], 
               kendalltau(x1, x2)[0],
               spearmanr(x1, x2)[0]]
        return float(abs(np.array(rho)).min())
    except: return 0

def _assign_woe(x, woe_df):

    '''
    Assign `WOE` with respect to BIN.
    
    Parameters
    ----------
    x : `pd.Series` object
    \t Input array of `float`.
    
    woe_df : `pd.Dataframe` object.
    \t `pd.DataFrame` with mandatory fields i.e. 'min', 
    \t 'max', 'bin', and 'woe'. `woe_df` is one of the 
    \t attributes from `evaluate_bins.woe_df`.
    
    Returns
    -------
    Array of `WOE`.
    '''
    if not isinstance(x, pd.Series):
        raise ValueError("'x' must be `pd.Series`. "
                         "Got {}".format(type(x)))
            
    if not isinstance(woe_df, pd.DataFrame):
        raise ValueError("'woe_df' must be `pd.DataFrame`. "
                         "Got {}".format(type(woe_df)))
    
    # Get BIN from `self.woe_df`.
    a = woe_df.loc[(woe_df['variable']==x.name)].copy()
    bins = a.loc[a['bin']>0,'min'].values.tolist() + [np.inf]
    woe = dict(n for n in woe_df[['bin','woe']].values)

    # `np.digitize()` assigns len(bins) to `np.nan`.
    # Since missing values in `woe_df` is in bin `0`,
    # such value has to be change accordingly.
    x = np.digitize(x, bins, right=False)                   
    x[x==len(bins)] = 0

    return np.array([woe[n] for n in x]).ravel()

def woe_transform(X, woe_df):

    '''
    ** woe_transform **
    
    Transform variables into `WOE` (Weight of Evidence)  
    according to bin intervals. The transformation will
    be carried out when all variable(s) in `X` match 
    exactly with `woe_df` in `variable` column. 
    
    .. versionadded:: 08-09-2020

    Parameters
    ----------
    X : `pd.DataFrame` object
    \t `pd.Dataframe` that will be tranformed into `WOE`. 
    
    woe_df : `pd.Dataframe` object.
    \t `pd.DataFrame` with mandatory fields i.e. 'variable', 
    \t 'min', 'max', 'bin', and 'woe'. `woe_df` is one of  
    \t the attributes from `evaluate_bins.woe_df`.

    Returns
    -------
    self.X : `pd.dataframe` object
    \t Transformed `X` with all `WOE`.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_breast_cancer as data
    >>> X, y = data(return_X_y=True)
    >>> X = pd.DataFrame(X, columns=[s.replace(' ','_') 
    ...                              for s in data().feature_names])

    # Use `woe_binning` to determine BIN.
    >>> binning = woe_binning(method='gini')
    >>> binning.fit(y, X['mean_texture'])
 
    # Use `evaluate_bins` to construct `woe_df`.
    >>> woe_table = evaluate_bins()
    >>> woe_table.fit(y, X['mean_texture'], binning.bin_edges)
 
    # Use `woe_transform` to transform to `WOE`.
    >>> woe_transform(X[['mean_texture']], woe_table.woe_df)
    '''
    if not isinstance(X, pd.DataFrame):
        raise ValueError("`X` must be `pd.DataFrame`. "
                         "Got {}".format(type(X))) 
        
    if not isinstance(woe_df, pd.DataFrame):
        raise ValueError("'woe_df' must be `pd.DataFrame`. "
                         "Got {}".format(type(woe_df)))
        
    # Match variables from `X` with `woe_df`.
    woe_var = np.unique(woe_df['variable'])
    columns = list(set(X.columns).intersection(woe_var))

    # Number of columns must match exactly with `woe_df`.
    if len(columns)==len(woe_var):
        a = [_assign_woe(X[var], woe_df).reshape(-1,1) 
             for var in columns]
        return pd.DataFrame(np.hstack(a),columns=columns)
    else: 
        raise ValueError("{:,} feature(s) found in `WOE`."
                         "{:,} match(es) found."
                         .format(len(woe_var),len(columns)))

class plot_woe:

    '''
    ** plot_woe **
    
    This functon helps user to visualize arrangement 
    of `WOE` (Weight of Evidence) given pre-determined 
    intervals, whereas missing or `np.nan` is binned 
    separately.
    
    Eaxmples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_breast_cancer as data
    
    >>> X, y = data(return_X_y=True)
    >>> columns = [s.replace(' ','_') for s in data().feature_names]
    >>> X = pd.DataFrame(X, columns=columns)

    # Use `woe_binning` to determine BIN.
    >>> binning = woe_binning(method='gini', equal_bin=5)
    >>> binning.fit(y, X['mean_texture'])
  
    # Use `evaluate_bins` to construct `woe_df`.
    >>> ev_model = evaluate_bins()
    >>> ev_model.fit(y, X['mean_texture'], binning.bin_edges)

    # Use `plot_woe` to visualize `evaluate_bins.woe_df`.
    >>> plot_woe(labels=data().target_names).plot(ev_model.woe_df)
    '''
    def __init__(self, **params):
        
        '''
        Parameters
        ----------
        **params : dictionary of properties, optional
        \t params are used to specify or override properties of 
    
            colors : `list` of color-hex codes
            \t List must contain 4 color-hex codes. The order is 
            \t the following i.e. positive-woe, negative-woe, 
            \t sample-percentage, and target-rate.

            labels : `list` of `str`
            \t `list` of labels whose items must be arranged in 
            \t ascending order. If `None`, it defaults to 
            \y ('Nonevent','Event').

            bardict : `dict` object
            \t A dictionary to override the default `ax.bar` 
            \t properties. 
            
            plotdict : `dict` object
            \t A dictionary to override the default `ax.plot` 
            \t properties.
            
            fontdict : `dict` object
            \t A dictionary to override the default `ax.annotate` 
            \t properties. 
            
            xticklabelsdict : `dict` object
            \t A dictionary to override the default 
            \t `ax.set_xticklabels` properties.
            
            legenddict : `dict` object
            \t A dictionary to override the default `ax.legend` 
            \t properties.
            
            float_format : `format()` method
            \t String formatting method for `float` e.g. 
            \t '{:.3g}'.format. This format is used in 
            \t `ax.set_xticklabels`.
            
            figsize : (`float`,`float`)
            \t Width, height in inches per plot. If `None`,
            \t it defautls to (5,4.5).
        '''
           
        def initialize_(name, params, default):
            if params.get(name) is not None:
                return params.get(name)
            else: return default
            
        def update_(name, params, default):
            if params.get(name) is not None:
                return {**default,**params.get(name)}
            else: return default   
            
        # =====================================================   
        # Get `colors` from `**params`.
        default = ['#25CCF7','#aaa69d','#f7f1e3','#eb2f06']
        self.colors = initialize_('color', params, default)
        # -----------------------------------------------------
        # Get `labels` from `**params`.
        self.labels = initialize_('labels', params, 
                                  ('Nonevent','Event'))
        # -----------------------------------------------------
        # Get `bardict` from `**params`.
        default = dict(alpha=0.7, width=0.7, align='center', 
                       edgecolor='#4b4b4b')
        self.bardict = update_('bardict', params, default)
        # -----------------------------------------------------
        # Get `plotdict` from `**params`.
        default = dict(marker='s', markersize=5, linewidth=1, 
                       linestyle='--', fillstyle='none')
        self.plotdict = update_('plotdict', params, default)
        # -----------------------------------------------------
        # Get `fontdict` from `**params`.
        default = dict(textcoords="offset points", ha='center', 
                       color='#4b4b4b', fontsize=10)
        self.fontdict = update_('fontdict', params, default)
        # -----------------------------------------------------
        # Get `xticklabelsdict` from `**params`.
        default = dict(fontsize=9, rotation=0)
        self.xticklabelsdict = update_('xticklabelsdict', 
                                       params, default)
        # -----------------------------------------------------
        # Get `legenddict` from `**params`.
        default = dict(loc='best', fontsize=10, framealpha=0, 
                       edgecolor='none')
        self.legenddict = update_('legenddict', params, default)
        # -----------------------------------------------------
        # Get `float_format` from `**params`.
        self.float_format = initialize_('float_format', params, 
                                        '{:.3g}'.format)
        # -----------------------------------------------------   
        # Get `figsize` from `**params`.
        self.figsize = initialize_('figsize', params, (5,4.5))
        # =====================================================

    def plot(self, woe_df, fname=None):

        '''
        Parameters
        ----------
        woe_df : `pd.DataFrame` object.
        \t `woe_df` is an attribute from 
        \t `evaluate_bins.fit.woe_df`.

        fname : `str` or `PathLike`, optional, default: None
        \t File path along with file name and extension (*.png).
        
        Returns
        -------
        Plots of 
        - Weight-of-Evidence 
        - Distribution of events & non-events
        - Target-rate and sample distribution
        '''
        columns = ['variable','min','max','bin','non_events',
                   'events','pct_nonevents','pct_events','woe','iv']
        
        if not isinstance(woe_df, pd.DataFrame):
            raise ValueError("`woe_df` must be `pd.DataFrame`. "
                             "Got {}".format(type(woe_df)))
        
        # Make sure all columns are there.
        self.df = woe_df.rename(str.lower,axis=1).copy()
        same = set(columns).intersection(self.df.columns)
        if len(same)!=len(columns):
            raise ValueError("`woe_df` must contain {} mandatory fields. "
                             "Got {}".format(len(columns),len(same)))
        
        # Get `variable` and its corresponding IV.
        self.var_name = self.df['variable'].values[0]  
        self.iv = self.df['iv'].sum()
        
        # Get X-axis tick-labels
        self.__ticklabels()
        
        # Plot all.
        size = (self.figsize[0]*3, self.figsize[1])
        fig, ax = plt.subplots(1, 3, figsize=size)
        
        # `WOE` plot.
        self._woe_plot(ax[0])
        
        # Distribution plot of two classes.
        self._distribution_plot(ax[1])
        
        # Target rate plot.
        self._target_plot(ax[2]) 
        
        if fname is not None: plt.savefig(fname)
        plt.tight_layout()
        plt.show()

    def __ticklabels(self):
        '''
        Set tick labels format (X-axis).
        '''
        ticklabels = np.empty(len(self.df), dtype='|U100')
        
        a = np.array([n for n in self.df['min'] if ~np.isnan(n)])
        labels = [r'$\geq$' + self.float_format(x) for x in a]
        self.xticklabels = ['\n'.join(('missing','(nan)'))] + labels
        
    def __set_ylimit(self, ax, ymin=None, ymax=None, factor=1.2):
        '''
        Set y-limit.
        '''
        t = ax.get_yticks()
        interval = np.diff(t)[0] * factor
        ymin = np.where(ymin==None, min(t)-interval, ymin)
        ymax = np.where(ymax==None, max(t)+interval, ymax)
        ax.set_ylim(float(ymin), float(ymax))        
            
    def _woe_plot(self, ax):
        '''
        Plot Weight-of-Evidence (`WOE`). 
        
        Parameters
        ----------
        ax : `axes.Axes` object
        \t `axes.Axes` object.
        
        Returns
        -------
        `axes.Axes` object`.
        '''
        # Extract positive and negative woes.
        woes = self.df['woe'].values
        pos = np.array([[n,w] for n,w in enumerate(woes) if w>=0])
        neg = np.array([[n,w] for n,w in enumerate(woes) if w<0])
        
        # Keyword arguments for vertical bar.
        posdict = dict(color=self.colors[0], label=r'%s $\geq$ %s'%self.labels)
        negdict = dict(color=self.colors[1], label=r'%s < %s'%self.labels)
        
        # Plot woes.
        ax.bar(pos[:,0], pos[:,1], **{**self.bardict, **posdict})
        ax.bar(neg[:,0], neg[:,1], **{**self.bardict, **negdict})
        
        # Keyword arguments for annotations.
        posdict = {**self.fontdict, **dict(va='bottom', xytext=(0,4))}
        negdict = {**self.fontdict, **dict(va='top', xytext=(0,-4))}
        
        # WOE values (annotation).
        s = '{:,.2f}'.format
        for xy in pos: ax.annotate(s(xy[1]), xy, **posdict)
        for xy in neg: ax.annotate(s(xy[1]), xy, **negdict)
        
        # Set y-label and x-tick-labels.
        ax.set_facecolor('white')
        ax.set_ylabel('Weight of Evidence (WOE)')
        ax.set_xticks(np.arange(len(woes)))
        ax.set_xticklabels(self.xticklabels, **self.xticklabelsdict)
    
        # Set title.
        title = ('Variable: {}'.format(self.var_name), 
                 'IV = {:,.4f} ({})'.format(self.iv, _iv_predict_(self.iv)))
        ax.set_title('\n'.join(tuple(title)))
        
        # Set `ax.set_ylim()`
        ymin,ymax = np.nanpercentile(woes,q=[0,100])
        gap = (ymax-ymin)*0.2
        ax.set_ylim(ymin-gap, ymax+gap)
        
        # Set legend.
        ax.legend(**self.legenddict)
        ax.grid(False)
        
        return ax
  
    def _distribution_plot(self, ax):
        '''
        Plot distribution of events and non-events.
        
        Parameters
        ----------
        ax : `axes.Axes` object
        \t `axes.Axes` object.
        
        Returns
        -------
        `axes.Axes` object`.
        '''                
        # Nonevents and Events
        pct = self.df[['pct_nonevents','pct_events']].values*100
        n_sample = self.df[['non_events','events']].values.sum(axis=0)
        
        # Width and offset of bars.
        width = 0.45          
        offset = width*0.5
        
        # Set labels.              
        float_format = ' ({:,.4g}, {:.0%})'.format             
        labels = [s + float_format(n,n/n_sample.sum()) 
                  for s,n in zip(self.labels,n_sample)]
     
        # Keyword arguments of bar.
        posdict = dict(color=self.colors[0], label=labels[0], width=width)
        negdict = dict(color=self.colors[1], label=labels[1], width=width)
                           
        # Plot distributions.
        x = np.arange(len(pct))
        ax.bar(x+offset, pct[:,0], **{**self.bardict, **posdict})
        ax.bar(x-offset, pct[:,1], **{**self.bardict, **negdict})
                           
        # Set annotation with respect to group (text).
        fontdict = {**self.fontdict, **dict(va='bottom', xytext=(0,4))}
        for n,p in enumerate(pct):
            ax.annotate('%d'%p[0], (n+offset, p[0]), **fontdict)
            ax.annotate('%d'%p[1], (n-offset, p[1]), **fontdict)
            
        # Set labels for both axes.
        ax.set_facecolor('white')
        ax.set_ylabel('Sample (%)')
        self.__set_ylimit(ax, ymin=0)
        ax.set_xticks(x)
        ax.set_xticklabels(self.xticklabels, **self.xticklabelsdict)
        
        # Set title.
        ax.set_title('Variable: {}\nSamples (%) ' 
                     'in each BIN'.format(self.var_name))

        # Set legend.
        ax.legend(**self.legenddict)
        ax.grid(False)
             
        return ax

    def _target_plot(self, ax):
        '''
        Plot distribution of samples and target rate.
        
        Parameters
        ----------
        ax : `axes.Axes` object
        \t `axes.Axes` object.
        
        Returns
        -------
        `axes.Axes` object`.
        '''
        # Number of samples and percentage by BIN.
        sample = self.df[['non_events','events']].values.sum(axis=1)
        p_sample = sample/sum(sample)*100
        
        # Number of targets and percentage by BIN.
        target = self.df['events'].values
        p_target = target/np.where(sample==0,1,sample)*100
          
        # Plot distribution and target rates.
        x = np.arange(len(sample))
        ax2 = ax.twinx()
   
        # Plot sample and target rate.
        bar = ax.bar(x, p_sample, **{**self.bardict, **{'color':self.colors[2]}})
        line = ax2.plot(x, p_target, **{**self.plotdict, **{'color':self.colors[3]}})
     
        # Set annotation for sample distribution.
        fontdict = {**self.fontdict, **dict(va='bottom', xytext=(0,4))}
        for n,s in enumerate(p_sample): ax.annotate('%d'%s, (n,s), **fontdict)
        
        # Set y-label and x-tick-labels.
        ax.set_facecolor('white')
        ax.set_ylabel('Sample (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(self.xticklabels, **self.xticklabelsdict)
        ax2.set_ylabel('Target Rate (%)', color=self.colors[3])

        # Set title.
        ax.set_title('Variable: {}\nTarget Rate (%)' 
                     ' in each BIN'.format(self.var_name))
        
        # Set `ylim` for both axes.
        self.__set_ylimit(ax, 0)
        self.__set_ylimit(ax2, 0)
        
        # Set labels.
        labels = ['Sample ({:,.4g})'.format(sample.sum()), 
                  'Target ({:,.4g}, {:,.0%})'.format(target.sum(),
                                                     target.sum()/sample.sum())]
        
        # Set legend.
        ax.legend([bar, line[0]],labels,**self.legenddict)
        ax.grid(False)

def _iv_predict_(iv):

    '''
    Information Value (IV) predictiveness
    It measures the strength of that relationship.
    '''
    if iv < 0.02: return 'Not useful for prediction'
    elif 0.02 <= iv < 0.1: return 'Weak predictive Power'
    elif 0.1 <= iv < 0.3: return 'Medium predictive Power'
    elif iv >= 0.3: return 'Strong predictive Power'   

