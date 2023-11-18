import random 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statannotations.Annotator import Annotator
from scipy.stats import mannwhitneyu, normaltest, wilcoxon
from statannot import add_stat_annotation


def seed_everything(seed):
    #Ensuring reproducible of training
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_training_curve(training_loss, validation_loss, training_f1, validation_f1, training_ap, validation_ap, epochs, name_curve):
    sns.set(style = "whitegrid")
    num_epochs = range(1, epochs + 1)
    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5), dpi=600)
    plt.plot(num_epochs, training_loss, label='Training Loss', linewidth=1)
    plt.plot(num_epochs, validation_loss, label='Validation Loss', linewidth=1)
    plt.plot(num_epochs, training_f1, label='Training f1 score', linewidth=1)
    plt.plot(num_epochs, validation_f1, label='Validation f1 score', linewidth=1)
    plt.plot(num_epochs, training_ap, label='Training AP', linewidth=1)
    plt.plot(num_epochs, validation_ap, label='Validation AP', linewidth=1)
    
    # Setting style for labels
    plt.title(name_curve, fontsize=16,fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=10)
    #plt.savefig("/content/drive/MyDrive/1_PROJECTS/D18/2. Chương/ALK/Chốt kết quả/ANN/training_history_test.png", bbox_inches='tight',dpi = 1200)
    plt.show()



def compare_opt_boxplot(baseline, optimize, metric, fig_name, save_path):
    AUC_internal = pd.DataFrame({"Baseline":baseline,
                     "Tune":optimize})


    df_melt = pd.melt(AUC_internal.reset_index(), id_vars=['index'], value_vars=AUC_internal.columns)
    df_melt.columns = ['index', 'Model', metric]
    subcat_palette = sns.dark_palette("#8BF", reverse=True, n_colors=5)
    
    stat_results = [wilcoxon(AUC_internal['Baseline'].astype('float'), AUC_internal['Tune'].astype('float'), alternative="two-sided"),
                    # wilcoxon(AUC_internal['PS'].astype('float'), AUC_internal['SS'].astype('float'), alternative="two-sided"),
                    # wilcoxon(AUC_internal['SS'].astype('float'), AUC_internal['GS'].astype('float'), alternative="two-sided"),
                    # wilcoxon(AUC_internal['GS'].astype('float'), AUC_internal['FS'].astype('float'), alternative="two-sided")
                   ]
    
    pvalues = [result.pvalue for result in stat_results]
    plotting_parameters = {
        'data':    df_melt,
        'x':       'Model',
        'y':       metric,
        'palette': subcat_palette[1:]
    }
    
    pairs = [('Baseline', 'Tune'),
             # ('PS', 'SS'),
             # ('SS', 'GS'),
             # ('GS', 'FS')
            ]
    
    
    sns.set_style("whitegrid")
    plt.figure(figsize = (10,7))
    
    my_colors = {'Tune': 'salmon', 
                 # 'PS': 'orange', 
                 # 'SS': 'lightgreen',
                 # 'GS': 'lightblue',
                 'Baseline': 'lightgreen'}
    
    
    ax = sns.boxplot(x='Model', y=metric, data=df_melt, palette=my_colors, showmeans=True ,meanprops={"marker":"d",
                               "markerfacecolor":"white", 
                               "markeredgecolor":"black",
                              "markersize":"5"})
    
    mean = round(AUC_internal.mean(),3)
    data = np.array(mean)   
    ser = pd.Series(data, index =AUC_internal.columns)
    
    dict_columns = {'Mean':mean,}
    df = pd.DataFrame(dict_columns)
    
    vertical_offset = df["Mean"].median()*0.008
    
    for xtick in ax.get_xticks():
        ax.text(xtick,ser[xtick]+ vertical_offset,ser[xtick], 
        horizontalalignment='center',color='k',weight='semibold', fontsize = 15)
    
    
    annotator = Annotator(ax, pairs, **plotting_parameters)
    annotator.configure(text_format="simple")
    annotator.set_pvalues_and_annotate(pvalues)
    
    ax.set_ylabel(metric, fontsize = 12)
    ax.set_xlabel(None)
    ax.set_xticklabels(labels = AUC_internal.columns, fontsize = 12)
    ax.set_title(fig_name,fontsize = 16, weight ='semibold')
    
    plt.savefig(save_path, dpi = 600)
    plt.show()