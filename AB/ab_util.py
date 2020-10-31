#-------------------------------------------------------------------------------------------------
# Utility functions for AB Analysis
# Author: Ui-Wing Cheah
#-------------------------------------------------------------------------------------------------

import pandas as pd 
import scipy.optimize as spopt
import numpy as np
import scipy as sp 
import seaborn as sns
import matplotlib.pyplot as plt

# set up the seaborn settings here
sns.set_context(context="paper",font_scale=1.5)
sns.color_palette('icefire',as_cmap=True)
sns.set_style('whitegrid')

#-------------------------------------------------------------------------------------------------
# Statistical convenience functions
#-------------------------------------------------------------------------------------------------
def get_cvar(input_series,lqtiles=[0.025,0.05,0.1,],rqtiles=[0.9,0.95,0.975]):
    """Computes tail-average (cvar if input_series is a series of returns)

    Args:
        input_series (pd.Series): the input series of values to compute the tail averages
        lqtiles (list, optional): left-tail breakpoints]. Defaults to [0.025,0.05,0.1,]
        rqtiles (list, optional): right-tail breakpoints]. Defaults to [0.9,0.95,0.975]

    Returns:
        cvar: Series with the quantile averages or the input series. The center average (10-90%tile) 
        is always included in the output
        
    """
    
    # getting the left-tail values
    lqvalues = input_series.quantile(lqtiles)
    cvar_l = pd.Series({x:input_series[input_series<=lqvalues[x]].mean() for x in lqtiles })
    
    # getting the right-tail values
    rqvalues = input_series.quantile(rqtiles)
    cvar_r = pd.Series({x:input_series[input_series>=rqvalues[x]].mean() for x in rqtiles })
    
    # get the central values here
    cvar_c = pd.Series({0.5:input_series[input_series>=lqvalues[0.1]][input_series<=rqvalues.loc[0.9]].mean()})
    
    # combine all values into one series
    cvar = cvar_l.combine_first(cvar_c).combine_first(cvar_r)
    return cvar


def get_sample_stats(input_series,):
    """Computes the 

    Args:
        input_series (Series/DataFrame): gets the conditional (percentile) means as well as the mean,median
        iqrange and standar deviation of the input series

    Returns:
        Series/DataFrame: returns the summary statistics
    """
    qtiles = get_cvar(input_series)
    disp = pd.Series({'mean':input_series.mean(), 'median':input_series.median(),
                      'iqrange':input_series.quantile(0.75)-input_series.quantile(0.25),
                      'stddev':input_series.std()})
    return disp.combine_first(qtiles)


def estimate_pvalue(sample,dist,*args,**kwargs):
    """computes the percentiles of the sample, by index

    Args:
        sample (Series): K-by-1 values of estimates to compare against the distribution
        dist (DataFrame): N-by-K dataframe of values. Note the DataFrame must have at least 
        all of the Series' indices in its columns.

    Returns:
        Series: Percentile values of input series relative to distribution
    """
    pvals = {}
    for j in sample.index.tolist():
        pvals[j] = sp.stats.percentileofscore(dist[j],sample[j])/100
    return pd.Series(pvals)


#-------------------------------------------------------------------------------------------------
# Plotting functions
#-------------------------------------------------------------------------------------------------
def box_scatter(sample_data,os_samples=[28,29,30],plot_cols ='tails',scatter_names = {},legend_dict={},xlabel=None,ylabel=None):
    """Code to do box-scatter plots"""
    os_samples = [28,29,30]
    col_renamer = {0:'0%tile',0.025:'2.5%tile',0.05:'5%tile',0.1:'10%tile',
                   0.9:'90%tile',0.95:'95%tile',.975:'97.5%tile',1.0:'100%tile',0.5:'10-90%tile'}

    # getting the summary statistics of the 3Y data
    sample_data = sample_data.rename(columns = col_renamer)

    if plot_cols=='tails':
        cols = ['2.5%tile','5%tile','10%tile','90%tile','95%tile','97.5%tile',]
    elif plot_cols=='center':
        cols = ['mean','median','10-90%tile']
    elif plot_cols=='disp':
        cols =['iqrange','stddev']
    else:
        cols = plot_cols

    # getting plot data
    plot_data = sample_data.reindex(columns=cols).stack()
    plot_data.index.names = ('sample','pct')
    plot_data = plot_data.reset_index().rename(columns={0:'value'})

    splot_data = sample_data.reindex(index=os_samples,columns=cols).rename(index=scatter_names).stack()
    splot_data.index.names = ('sample','pct')
    splot_data = splot_data.reset_index().rename(columns={0:'value'})

    ax_ = sns.boxplot(data=plot_data,x='pct',y='value',)
    ax_ = sns.scatterplot(data=splot_data,y='value',x='pct',hue='sample',ax=ax_,palette='dark',s=140,alpha=0.5)
    ax_.legend(**legend_dict)
    if xlabel is not None:
        ax_.set_xlabel(xlabel)
    if ylabel is not None:
        ax_.set_ylabel(ylabel)
    return ax_


#-------------------------------------------------------------------------------------------------
# Bootstrapping convenience functions
# Not sure if we will use these
#-------------------------------------------------------------------------------------------------
def bootstrap_by_year(data_df,ndraws=1000,ksample=10,insample=True,*args,**kwargs):
    """ bootstrapping function 
        uniform distribution assumed for random sampling
    """
    # set the random seed so we can recreate the data
    np.random.seed(100)
    if insample:
        choice_years = sorted(set(data_df[data_df['sample']==0]['year'].values))
    else:
        choice_years = sorted(set(data_df['year'].values))

    # looping through the return series and concatenate
    bootstrap_sample = {}
    for i in range(ndraws):
        # select random vector of ksample-length from the choice set
        ix_rand = np.random.choice(choice_years,size=ksample)
        rets_random = np.concatenate([data_df[data_df['year']==j]['returns'].values for j in ix_rand])
        bootstrap_sample[i] = pd.Series(data=rets_random)
    
    bootstrap_sample=pd.DataFrame(bootstrap_sample)
    return bootstrap_sample

def get_bootstrap_stats(btstrp,):
    # qtiles = btstrp.quantile([0.,0.025,0.05,0.95,.975,1.])
    qtiles = pd.DataFrame({i:get_cvar(btstrp[i]) for i in btstrp.columns})
    disp = pd.DataFrame({'mean':btstrp.mean(), 'median':btstrp.median(),
    'iqrange':btstrp.quantile(0.75)-btstrp.quantile(0.25),
    'stddev':btstrp.std()})
    return disp.combine_first(qtiles.T)
