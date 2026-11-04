"""Create plots from simulation arrays; importing this module does not run a simulation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from simulation import SIGNALS


def save_plots(result, reference, output):
    t,y=result['t'],result['y']
    styles={'python':'#c72b2b','reference':'#1669b5'}
    def decorate(ax,title,label):
        ax.set(title=title,xlabel='Time (s)',ylabel=label,xlim=(0,10))
        ax.grid(alpha=.22)
        ax.legend(fontsize=7,loc='best')
    def ltr(ax,i):
        ax.plot(t,y[:,i],color='black',lw=1.5,label='Python passive')
        ax.plot(t,y[:,i+2],color=styles['python'],lw=1.5,label='Python LQR1')
        r=reference[SIGNALS[i]];ax.plot(r[:,0],r[:,1],ls='--',color='.45',lw=1.3,label='Reference passive')
        r=reference[SIGNALS[i+2]];ax.plot(r[:,0],r[:,1],ls='--',color=styles['reference'],lw=1.3,label='Reference LQR1')
        ax.axhline(-1,color='.4',lw=.8,ls=':')
        decorate(ax,'Front axle' if i==0 else 'Rear axle',r'$R_f$' if i==0 else r'$R_r$')
    def current(ax,i):
        ax.plot(t,1000*y[:,i+4],color=styles['python'],lw=1.5,label='Python LQR1')
        r=reference[SIGNALS[i+4]]
        ax.plot(r[:,0],1000*r[:,1],ls='--',color=styles['reference'],lw=1.3,label='Reference LQR1')
        decorate(ax,'Front actuator' if i==0 else 'Rear actuator',r'$u_f$ (mA)' if i==0 else r'$u_r$ (mA)')
    def save(fig,name):
        for suffix in ['png','pdf']:
            fig.savefig(output/(name+'.'+suffix),dpi=200,facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for i in range(2):
        ltr(axes[0,i],i);current(axes[1,i],i)
    for label,ax in zip(['(a)','(b)','(c)','(d)'],axes.flat):
        ax.set_title(label+' '+ax.get_title())
    save(fig,'Figure1_time_responses')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for i,ax in enumerate(axes): ltr(ax,i)
    save(fig,'time_Rf_Rr')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for i,ax in enumerate(axes): current(ax,i)
    save(fig,'time_uf_ur')
    fig,ax=plt.subplots(figsize=(8,3),layout='constrained')
    ax.plot(t,np.rad2deg(result['delta']),color='black')
    ax.set(xlabel='Time (s)',ylabel='Steering angle (deg)',xlim=(0,10))
    ax.grid(alpha=.22)
    save(fig,'steering_input')