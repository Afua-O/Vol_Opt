import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates #replace pandas.tools.plotting
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.colors as mpl

sns.set_style('whitegrid')

def truncate_cmap (cmap,n_min=0,n_max=256):
    """ Generate a truncated colormap 
    Source for this funciton: https://gist.github.com/astrodsg/09bfac1b68748967ed8b#file-mpl_colormap_tools
        
    """
    color_index = np.arange(n_min,n_max).astype(int)
    colors = cmap(color_index)
    name = "truncated_{}".format(cmap.name)
    return plt.matplotlib.colors.ListedColormap(colors,name=name)

def parallel_plots(input_folder,input_file_name,output_folder,plot_title):

    names=['Hydropower','Irrigation','Environment', 'Flood_control']
    data = pd.read_csv(input_folder+input_file_name+'.csv', usecols=['annualhydropower','irrigation','environment', 'floodcontrol'])
    mn_mx= pd.read_csv(input_folder+'min_max.csv', usecols=names)
    units=['Energy [MWh/year]','Met_irri_demand ','[E-flow_reliability]','[Flood protection]']
    nobjs=4
    policies=5 # number of extreme policies(one for each objective) + compromise policy


    mx=[]
    mn=[]
    for i in range(len(names)):
        if i ==2: 

            mini=str(round(mn_mx[names[i]][1],3))
            maxi=str(round(mn_mx[names[i]][0],3))
            mx.append(maxi)
            mn.append(mini)


        else:
    	    mini=str(round(mn_mx[names[i]][1],1))
    	    maxi=str(round(mn_mx[names[i]][0],1))
    	    mx.append(maxi)
    	    mn.append(mini)


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    objs1=data['annualhydropower'] # here you choose the objective used for the colormap

    gray='#bdbdbd'; purple='#7a0177'; green='#41ab5d'; blue='#1d91c0';yellow='#fdaa09'; pink='#c51b7d' #HEX
    

    #### grey scale
    # cmap=truncate_cmap(plt.cm.Greys,n_min=20,n_max=120) # truncate to avoid white and black lines
    # colors=cmap(objs1)

    #find the position where each objective's value is 1
    # l=len(data)

    # colors[l-policies,:]=mpl.to_rgba_array(purple)
    # colors[l-policies+1,:]=mpl.to_rgba_array(yellow)
    # colors[l-policies+2,:]=mpl.to_rgba_array(blue)
    # colors[l-policies+3,:]=mpl.to_rgba_array(green)
    # colors[l-policies+4,:]=mpl.to_rgba_array(pink)
    

    # for a simpler approach with plain gray instead of colormap of grays:
    colors=[gray,purple,green,blue,yellow,pink]


    parallel_coordinates(data,'floodcontrol',color=colors,linewidth=10)


    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
     #      ncol=3, mode="expand", borderaxespad=1.5, fontsize=18) 

    i=0
    ax1.set_xticks(np.arange(nobjs))
    ax1.set_xticklabels([mx[i]+'\n'+names[i], mx[i+1]+'\n'+names[i+1],mx[i+2]+'\n'+names[i+2],mx[i+3]+'\n'+names[i+3]],fontsize=40)
    ax2 = ax1.twiny()
    ax2.set_xticks(np.arange(nobjs))
    ax2.set_xticklabels([mn[i], mn[i+1],mn[i+2],mn[i+3]], fontsize=40)
    ax1.get_yaxis().set_visible([])
    ax1.get_legend().remove()
    plt.text(-.05, 0.5, ' Direction of Preference  $\\longrightarrow$', {'color': '#636363', 'fontsize':  40},
             horizontalalignment='left',
             verticalalignment='center',
             rotation=90,
             clip_on=False,
             transform=plt.gca().transAxes)
    plt.legend('False') # remove legend

    # plt.colorbar.ColorbarBase(ax=ax, cmap=cmap,
    #                              orientation="vertical")
    fig.set_size_inches(17.5, 9)
    return(plt.savefig(output_folder+'pdf/'+plot_title+'.pdf', bbox_inches="tight", transparent=True), plt.savefig(output_folder+'png/'+plot_title+'.png', bbox_inches="tight", transparent=True))

plot_folder="../plots/"
input_folder="../for_plots/"
plot_title='1984 50k nfe'
input_file_name='10_solution'

if not os.path.exists(plot_folder+'pdf'):
    os.makedirs(plot_folder+'pdf')

if not os.path.exists(plot_folder+'png'):
    os.makedirs(plot_folder+'png')





parallel_plots(input_folder,input_file_name, plot_folder,plot_title)



