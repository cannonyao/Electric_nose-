
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.gridspec as gridspec


## Defining input files and times

# banana_id = 5
# wine_id   = 0
b_max_threshold = 1
b_min_threshold = -0.5
w_max_threshold = 1.5
w_min_threshold = -0.5


## Importing metadata and induction information
metadata = np.loadtxt('HT_Sensor_metadata.dat', skiprows=1, dtype=str)
metadata_aux = np.array( metadata[:,[0,3,4]], dtype=float )
banana_ids = metadata[metadata[:,2] == 'banana' , 0]
wine_ids   = metadata[metadata[:,2] == 'wine' , 0]
i = 0

for banana_id , wine_id in zip(banana_ids,wine_ids):

    print("process %d" % i)
    i = i + 1
    banana_info = metadata_aux[int(banana_id)]
    bt0 = 0
    btf = banana_info[2]

    wine_info   = metadata_aux[int(wine_id)]
    wt0 = 0
    wtf = wine_info[2]


    ## Loading the dataset
    dataset = np.loadtxt('HT_Sensor_dataset.dat', skiprows=1)

    bData = dataset[dataset[:,0] == int(banana_id),1:]
    bData = bData[bData[:,0] <= b_max_threshold]
    bData= bData[bData[:,0] >= b_min_threshold]
    #bData[:,0] += btmin_threshold <= 0

    wData = dataset[dataset[:,0] == int(wine_id),1:]
    wData = wData[wData[:,0] <= w_max_threshold]
    wData = wData[wData[:,0] >= w_min_threshold]
    #wData[:,0] += wt0




    ## Starting the plot

    pl.figure(figsize = (9,7))
    gs1 = gridspec.GridSpec(6, 2  )
    gs1.update(wspace=0.4, hspace=0.0)

    ax = {}
    for j in range(6):
        ax[j,0] = pl.subplot(gs1[j*2])
        ax[j,1] = pl.subplot(gs1[j*2+1])



    ## Plotting all data

    ax[0,0].plot( bData[:,0], bData[:,10], color=(0.1,0.8,0.1), lw=1.5 )
    ax[0,0].set_ylim(45,70)
    ax[0,1].plot( wData[:,0], wData[:,10], color=(0.1,0.8,0.1), lw=1.5 )
    ax[0,1].set_ylim(45,70)
    ax[0,0].set_yticks( np.arange(56.,65.5,3) )
    ax[0,1].set_yticks( np.arange(56.,65.5,3) )

    ax[1,0].plot( bData[:,0], bData[:, 9], color=(1.0,0.1,0.0), lw=1.5 )
    ax[1,0].set_ylim(20,35)
    ax[1,1].plot( wData[:,0], wData[:, 9], color=(1.0,0.1,0.0), lw=1.5 )
    ax[1,1].set_ylim(20,35)
    ax[1,0].set_yticks( np.arange(27.,29.5,1) )
    ax[1,1].set_yticks( np.arange(27.,29.5,1) )

    ax[2,0].plot( bData[:,0], bData[:, 1], color=(0.3,0.3,0.3), lw=1.5)
    ax[2,0].plot( bData[:,0], bData[:, 4], '-', color=(1.0,0.5,0.1), lw=1.5 )
    ax[2,0].set_ylim(7.2,13.9)
    ax[2,1].plot( wData[:,0], wData[:, 1], color=(0.3,0.3,0.3), lw=1.5, label=r"$R_1$"  )
    ax[2,1].plot( wData[:,0], wData[:, 4], color=(1.0,0.5,0.1), lw=1.5, label=r"$R_4$"  )
    ax[2,1].set_ylim(7.2,13.9)
    ax[2,0].set_yticks( np.arange(8.,12.5,2) )
    ax[2,1].set_yticks( np.arange(8.,12.5,2) )

    ax[2,1].legend(frameon = False, fontsize=12, bbox_to_anchor=(-0.08,0.9), handletextpad=0)

    ax[3,0].plot( bData[:,0], bData[:, 2], '--', color=(0.3,0.3,0.3), lw=1.5, zorder=3 )
    ax[3,0].plot( bData[:,0], bData[:, 3], color=(1.0,0.5,0.1), lw=1.5, zorder=1 )
    ax[3,0].set_ylim(3.2,12.9)
    ax[3,1].plot( wData[:,0], wData[:, 2], color=(0.3,0.3,0.3), lw=1.5, label=r"$R_2$" )
    ax[3,1].plot( wData[:,0], wData[:, 3], color=(1.0,0.5,0.1), lw=1.5, label=r"$R_3$"  )
    ax[3,1].set_ylim(3.2,12.9)
    ax[3,0].set_yticks( np.arange(5.,12.5,2 ) )
    ax[3,1].set_yticks( np.arange(5.,12.5,2) )

    ax[3,1].legend(frameon = False, fontsize=12, bbox_to_anchor=(-0.08,0.9), handletextpad=0)

    ax[4,0].plot( bData[:,0], bData[:, 5], '-', color=(0.3,0.3,0.3), lw=1.5 )
    ax[4,0].plot( bData[:,0], bData[:, 6], '-', color=(1.0,0.5,0.1), lw=1.5 )
    ax[4,0].set_ylim(3.0,15)
    ax[4,1].plot( wData[:,0], wData[:, 5], '-', color=(0.3,0.3,0.3), lw=1.5, label=r"$R_5$" )
    ax[4,1].plot( wData[:,0], wData[:, 6], '-', color=(1.0,0.5,0.1), lw=1.5, label=r"$R_6$" )
    ax[4,1].set_ylim(3.,15)
    ax[4,0].set_yticks( np.arange(4.,12.5,4) )
    ax[4,1].set_yticks( np.arange(4.,12.5,4) )

    ax[4,1].legend(frameon = False, fontsize=12, bbox_to_anchor=(-0.08,0.9), handletextpad=0)

    ax[5,0].plot( bData[:,0], bData[:, 7], '-', color=(0.3,0.3,0.3), lw=1.5 )
    ax[5,0].plot( bData[:,0], bData[:, 8], '-', color=(1.0,0.5,0.1), lw=1.5 )
    ax[5,0].set_ylim(1.1,6.8)
    ax[5,1].plot( wData[:,0], wData[:, 7], '-', color=(0.3,0.3,0.3), lw=1.5, label=r"$R_7$" )
    ax[5,1].plot( wData[:,0], wData[:, 8], '-', color=(1.0,0.5,0.1), lw=1.5, label=r"$R_8$" )
    ax[5,1].set_ylim(1.1,6.8)
    ax[5,0].set_yticks( np.arange(2.,6.5,2) )
    ax[5,1].set_yticks( np.arange(2.,6.5,2) )

    ax[5,1].legend(frameon = False, fontsize=12, bbox_to_anchor=(-0.08,0.9), handletextpad=0)



    ## Setting up limits and ticks

    for j in range(6):
        ax[j,0].plot( [bt0,bt0,btf,btf], [-100,100,100,-100], '-',
                      color=(0.1,0.1,1.0), lw=2.0, alpha = 0.5 )
        ax[j,1].plot( [wt0,wt0,wtf,wtf], [-100,100,100,-100], '-',
                      color=(0.1,0.1,1.0), lw=2.0, alpha = 0.5 )
        #x[j,0].set_xticks(np.arange(6.0,8,0.25))
        #ax[j,0].set_xlim(6.0, bData[:,0].max())
        #ax[j,0].set_xticklabels([])
        #ax[j,1].set_xticks(np.arange(22,23.8,0.25))
        #ax[j,1].set_xlim(wData[:,0].min(), wData[:,0].max())
        #ax[j,1].set_xticklabels([])
        #ax[j,1].xaxis.grid()



    ## Writing labels

    ax[0,0].set_title("Banana")
    ax[0,1].set_title("Wine")

    ax[0,0].set_ylabel(r"$H$ (%)")
    ax[1,0].set_ylabel(r"$T_E$ (C)")
    ax[2,0].set_ylabel(r"$R_{1,4}$ (k$\Omega$)")
    ax[3,0].set_ylabel(r"$R_{2,3}$ (k$\Omega$)")
    ax[4,0].set_ylabel(r"$R_{5,6}$ (k$\Omega$)")
    ax[5,0].set_ylabel(r"$R_{7,8}$ (k$\Omega$)")

    # ax[5,0].set_xticklabels(['6.0','','6.5','','7.0','','7.5'])
    # ax[5,1].set_xticklabels(['22.0','','22.5','','23.0','','23.5'])
    ax[5,0].set_xlabel("Time (h)")
    ax[5,1].set_xlabel("Time (h)")


    ## Saving the plot as a png file
    pl.savefig("%d.png" % i, dpi=300)




