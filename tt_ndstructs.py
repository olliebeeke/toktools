# Classes that contain 0d, 1d and 2d data objects.

from jet.data import sal
import numpy as np
from copy import deepcopy as cp

class zerod_item:

    def __init__( self, signal, data, tag, description = "", units = "", sym = ""):
        self.tag = tag
        self.signal = signal
        if description != "":
            self.description = description
        else:
            self.description = signal.description
        if units != "":
            self.units = units
        else:
            self.units = '[' + signal.units + ']'
        self.data = data
        self.sym = sym

    def normalize(self, normFac, normSym=""):
        self.data = self.data/normFac
        self.units = ""
        self.sym = normSym

class oned_item(zerod_item):
    def __init__( self, signal, data, tag, ydescription = "", yunits = "", ysym = ""):
        super().__init__(signal, data, tag, description = ydescription, units = yunits, sym=ysym)
        self.xdescription = signal.dimensions[0].description
        self.xunits = signal.dimensions[0].units
        self.xdata = signal.dimensions[0].data

    def reduce_dim ( self, value_choice, interpolate ):
        dimDat = self.signal.dimensions[0].data
        idat = int((np.abs(dimDat-value_choice)).argmin())
        # Using slice objects so that we can generalize this for >1-d arrays.
        sl1 = [slice(None)] * self.data.ndim
        if not interpolate or dimDat[idat] == value_choice:
            # If not interpolating (or the time matches exactly), just find data at time index closest to target time.
            sl1[0] = slice(idat,idat+1)
            data = self.data[sl1]
        else:
            # Linearly interpolate between the two closest grid points.
            sl2 = [slice(None)] * self.data.ndim

            if dimDat[idat] < value_choice and idat + 1 < len(dimDat):
                sl1[0] = slice(idat,idat+1)
                sl2[0] = slice(idat+1,idat+2)
            elif dimDat[idat] > value_choice and idat != 0:
                sl1[0] = slice(idat-1,idat)
                sl2[0] = slice(idat,idat+1)
            else:
                print("!! Error. Cannot extrapolate beyond available dataset. Available data are in the range {:1.2f} to {:1.2f} {}. Exiting !!".format(dimDat[0],dimDat[-1], self.signal.dimensions[0].units))
                quit()
            m = (self.data[sl2]-self.data[sl1]) / (dimDat[sl2[0]]-dimDat[sl1[0]])
            c = self.data[sl1] - m*dimDat[sl1[0]]
            data = m*value_choice + c
            #print("Value at nearest data point: {:1.2f}".format(self.data[sl1]))
            #print("Interpolated value: {:1.2f}".format(data))
        del self.signal.dimensions[0]
        if isinstance(self, twod_item):
            return oned_item( self.signal, data[0], self.tag, ysym = self.sym)
        else:
            return zerod_item( self.signal, data, self.tag , sym = self.sym)

class twod_item(oned_item):
    def __init__( self, signal, data, tag, zdescription = "", zunits = "", zsym = ""):
        super().__init__(signal, data, tag, ydescription = zdescription, yunits = zunits, ysym = zsym)
        self.ydescription = signal.dimensions[1].description
        self.yunits = signal.dimensions[1].units
        self.ydata = signal.dimensions[1].data
        
