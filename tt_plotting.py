import numpy as np
import matplotlib
matplotlib.use("PDF")
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from PyPDF2 import PdfFileMerger, PdfFileReader
import os

# setup some plot defaults OB 200918 ~ hacky solution to prevent latex drawing on archer (cray system).
plt.rc('text', usetex='cray' not in os.environ['MANPATH'])
plt.rc('font', family='serif')
plt.rc('font', size=30)
rcParams.update({'figure.autolayout': True})
rcParams.update({'legend.fontsize': 20, 'legend.handlelength': 4})
#rcParams.update({'legend.frameon': False})


myred = [183./255, 53./255, 53./255]
myblue = [53./255, 118./255, 183./255]
oxblue = [0.,33./255,71./255]
oxbluel = [68./255,104./255,125./255]
oxbluell = [72./255,145./255,220./255]

def save_plot(pdfname):  
    if ifile is not None:
        filename = run.work_dir + run.dirs[ifile] + run.out_dir + pdfname + '_' + run.files[ifile] + '.pdf'
    else:
        filename = run.work_dir + pdfname + '.pdf'
    plt.savefig(filename, bbox_inches='tight')

def merge_pdfs(in_namelist, out_name):

    # Read PDFs to be merged.
    merger = PdfFileMerger()
    for pdfname in in_namelist:
        file_name = pdfname
        with open(file_name, 'rb') as pdffile:
            merger.append(PdfFileReader(pdffile))

    # Merge PDFs.
    merger.write(out_name)

    # Remove constituent PDFs
    for pdfname in in_namelist:
        os.system('rm -f '+file_name)

    plt.cla()
    plt.clf()

def set_plot_defaults():
    import os
    # setup some plot defaults OB 200918 ~ hacky solution to prevent latex drawing on archer (cray system).
    plt.rc('text', usetex='cray' not in os.environ['MANPATH'])
    plt.rc('font', family='serif')
    plt.rc('font', size=30)
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'legend.fontsize': 20, 'legend.handlelength': 4})
    rcParams.update({'legend.frameon': False})

def plot_1d(x,y,xlab, axes = None, title='',ylab='', label = "", color = '#0099aa', linestyle = '-', grid="both", marker='None', markersize=5.0, errors = [], linewidth=1.5, log="", markxpos=0.0):
    if axes == None:
        fig = plt.figure(figsize=(12,8))
        axes = plt.gca()
    # Draw a vertical line from the x axis to the line.
    if markxpos > 0.0:
        i = int((np.abs(x-markxpos)).argmin())
        if x[i] < markxpos and i + 1 < len(x):
            i1 = i
            i2 = i+1
        elif x[i] > markxpos and i != 0:
            i1 = i-1
            i2 = i
        m = (y[i2]-y[i1]) / (x[i2]-x[i1])
        c = y[i1] - m*x[i1]
        ymark = m*markxpos + c
        axes.plot([markxpos,markxpos], [0.0, ymark], linewidth=2, linestyle=':', color='gray')
        axes.plot([0,markxpos], [ymark, ymark], linewidth=2, linestyle=':', color='gray')
        axes.scatter(markxpos, ymark, color=color)
        axes.set_xlim(np.amin(x), np.amax(x))
    if len(errors) > 0:
        axes.errorbar(x,y,yerr=errors,label=label, color = color, linestyle = linestyle, marker=marker, markersize=markersize, capsize=2.5, lw = linewidth)
    else:
        axes.plot(x,y,label=label, color = color, linestyle = linestyle, marker=marker, markersize=markersize, lw = linewidth)
    axes.set_xlabel(xlab)
 
    if log == "x":
        axes.set_xscale('log')
    elif log == "y":
        axes.set_yscale('log')
    elif log == "both":
        axes.set_xscale('log')
        axes.set_yscale('log')

    if len(ylab) > 0:
        axes.set_ylabel(ylab)
    if len(title) > 0:
        axes.set_title(title)
    if len(grid) >0:
        axes.grid(b=True, axis=grid)

def plot_multi_1d(x,y,xlab, axes=None, labels=[],legendtitle="", title='', ylab = '', grid="", log = "", errors = []):
    if axes == None:
        fig = plt.figure(figsize=(12,8))
        axes = plt.gca()

    colors = truncate_colormap(minval = 0.0, maxval = 0.9, n = len(y))

    if log == "x":
        plt.xscale('log')
    elif log == "y":
        plt.yscale('log')
    elif log == "both":
        plt.xscale('log')
        plt.yscale('log')
     
    if np.ndim(y)>1:
        for i in range(len(y)):
            print(y[i])
            print(labels)
            print(labels[i])
            if len(errors) > 0:
                plot_1d(x,y[i],xlab, axes=axes, color=colors(i), label=labels[i], errors=errors[i])
            else:
                plot_1d(x,y[i],xlab, axes=axes, color=colors(i), label=labels[i])
    else:	# Plotting just one variable.
        print('Just one variable, should use plot_1d not plot_multi_1d! Plot cancelled.')

    plt.legend(ncol=1,title=legendtitle, bbox_to_anchor=(1.02, 1.05), handlelength=1.5)
    
    if len(ylab) > 0:
        plt.ylabel(ylab)
    if len(title) > 0:
        plt.title(title)
    if len(grid) > 0:
        plt.grid(b=True, axis=grid)

# OB ~ Edited to take into account non-uniform grids.
def plot_2d(z,xin,yin,zmin,zmax,xlab='',ylab='',title='',cmp='RdBu',use_logcolor=False, interpolation='nearest', markersize=[], anim=False):
    from scipy.interpolate import griddata
    if not anim:
        fig = plt.figure(figsize=(12,8))
    nxin,nyin = len(xin),len(yin)
    last_xin, last_yin = xin[nxin-1], yin[nyin-1]
    # Z is likely provided in a 2-d array format. We need to convert this to a single row of length len(xin) * len(yin). TODO CHECK IF Z ALWAYS IN 2D ARRAY FORMAT.
    data_xy = np.zeros((nxin*nyin,2))
    data_values = np.zeros(nxin*nyin)
    for ix in range(nxin):
        for iy in range(nyin):
            data_xy[nxin*iy + ix, 0] = xin[ix]
            data_xy[nxin*iy + ix, 1] = yin[iy]
            data_values[nxin*iy + ix] = z[ix,iy]

    # Generate fine grid on which to interpolate.
    x=np.linspace(xin[0],last_xin,max(1000,len(xin)))
    y=np.linspace(yin[0],last_yin,max(1000,len(yin)))
    grid_x, grid_y = np.meshgrid(x,y)           
    if use_logcolor:
        color_norm = mcolors.LogNorm(zmin,zmax)
    else:
        color_norm = mcolors.Normalize()
    z_interp = griddata(data_xy,data_values,(grid_x,grid_y),method=interpolation)
    im = plt.imshow(z_interp,cmp, 
                #vmin=np.amin(data_values), vmax=np.amax(data_values),
                vmin=zmin,vmax=zmax,
                extent=[x.min(),x.max(),y.min(),y.max()],
                interpolation='nearest', origin='lower', aspect='auto', norm=color_norm, animated=anim)
    if anim:
        return im
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plotax = plt.gca()
    plt.colorbar()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    if 30 in markersize: # Add markers to scatter plot and some extra legend to denote what they mean.
        markersize = np.array([30]*nxin*nyin)
        nonzero_area = np.ma.masked_where(data_values==0, markersize)
        zero_area = np.ma.masked_where(data_values>0, markersize)
        plt.scatter(*np.meshgrid(xin,yin), s=zero_area, marker='o', label='$\\rm{Indicates\\ where}\\ \\gamma_{\\rm{max}}\\ \\rm{occurs\\ at\\ largest\\ } k_y$')
        plt.legend()
        # Shrink current axis's height by 10% on the bottom
        box = plotax.get_position()
        #plotax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
        # Put a legend below current axis
        plotax.legend(loc='upper left', bbox_to_anchor=(-0.1, -0.16))	
        
    
    return (fig,im)

def truncate_colormap(name='nipy_spectral', minval = 0.0, maxval = 1.0, n = 256):
    to_truncate = matplotlib.cm.get_cmap(name, 256)
    new_colors = to_truncate(np.linspace(minval,maxval, n))
    return mcolors.ListedColormap(new_colors)


def movie_2d(z,xin,yin,zmin,zmax,nframes,outfile,xlab='',ylab='',title='',step=1,cmp='RdBu'):

    from matplotlib import animation

    fig = plt.figure(figsize=(12,8))
    x,y = np.meshgrid(xin,yin)
    im = plt.imshow(z[0,:,:], cmap=cmp, vmin=zmin, vmax=zmax,
               extent=[x.min(),x.max(),y.min(),y.max()],
               interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    
    ims = []

    for i in range(1,nframes,step):
        im = plt.imshow(z[i,:,:], cmap=cmp, vmin=zmin, vmax=zmax,
               extent=[x.min(),x.max(),y.min(),y.max()],
               interpolation='nearest', origin='lower', aspect='auto')
        ims.append([im])

    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
    ani.save(outfile)

def movie_1d(x,y,xmin,xmax,ymin,ymax,nframes):

    from matplotlib import animation

    fig = plt.figure(figsize=(12,8))
    ax=plt.axes(xlim=(xmin,xmax),ylim=(ymin,ymax))
    line, = ax.plot([],[],lw=2)

    def init():
        line.set_data([],[])
        return line,

    def animate(i):
        line.set_data(x,y[i,:])
        return line,

    anim=animation.FuncAnimation(fig, animate, init_func=init, 
                                 frames=nframes, interval=200)

    return anim

