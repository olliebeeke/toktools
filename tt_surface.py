from jet.data import sal
import tt_ndstructs as tt_structs
import tt_plotting as tt_plot
import numpy as np
from copy import deepcopy as cp
import matplotlib.pyplot as plt
import f90nml

# This script uses the SAL library to access JET Data. To access this remotely, you need to have remote access to JET. 
# The username is your JET username: replace the default below with your username.
# The password uses the RSA keyfob with two-factor authentication: Enter your PIN (4 digits) followed by the 6-digit code from the keyfob.
# e.g. if your PIN is 1234, and the keyfob shows 098765, then your password is 1234098765.

class surfobj:

    def __init__( self, source, shot, uid, seq, t_choice, interpolate = False, SALuser = ''):
        if SALuser == '':
            print('  !!! You can set your remote access username via SALuser, in tt_main.py !!!')
        sal.authenticate(SALuser)   # Enter your username as the argument here, or leave no argument if you want to enter your username every time. 
        self.shot = shot
        self.uid = uid
        self.t_choice = t_choice
        ( self.zerod, self.oned, self.twod ) = self.get_attr( source, shot, uid, seq, t_choice, interpolate )

    def get_attr( self, source, shot, uid, seq, t_choice, interpolate ):
        # Set up stem for data-access.
        defaultStem = '/pulse/{}/ppf/signal/{}'.format(shot, uid)
        branchInfo = sal.list(defaultStem)
        if seq == -1:
            seq = branchInfo.revision_latest
            self.seq = seq
        if seq in branchInfo.revision_modified:
            # Check whether sequence number exists for this UID and shot.
            seq_post = ':{}'.format(seq)
        else:
            print('Could not find sequence {} for pulse data in {}'.format(seq,defaultStem))
            quit()
        print('Getting data from {} for sequence {}.'.format(defaultStem,seq))
        
        # Get one-dimensional variables.
        # TODO check whether this FBND is actually used!
        # Signal code
        od_signals = ['fbnd','btor']
        # Signal uid; some signals stored in jetppf instead of our 'desired' uid.
        od_uids = ['jetppf',uid]
        # Signal sequence; take the most recent sequence for non-uid signals.
        od_seq_posts = ['',seq_post]
        # Signal DDA; The folder where we will find the signal. e.g. jsp, prfl, efit etc...
        od_ddas = ['efit','jst']
        # Signal tag; Used to search for the signal later on once it is stored in a dictionary.
        od_tags = ['phia','1d_template']
        
        zerod = {}
        oned = {}
        twod = {}

        for i in range(len(od_signals)):
            if od_uids[i] != '':
               sigString = '/pulse/{}/ppf/signal/{}/{}/{}{}'.format(shot, od_uids[i], od_ddas[i], od_signals[i],od_seq_posts[i])
            else:
                sigString = defaultStem + '/{}/{}'.format(od_ddas[i], od_signals[i])
            try:
                node = sal.get(sigString)
                print('Found {} at {}.'.format(od_tags[i], sigString))
                oned[od_tags[i]] = tt_structs.oned_item(node, node.data, od_tags[i])
            except:
                print('! --- {} not found at {} --- !'.format(od_tags[i], sigString))
                continue
         
        # Get two-dimensional variables.
        td_signals=['te','ti','ne','nih','nid','nit','nim1','nim2','nim3','nimp','zia1','zia2','zia3','dnbd','drfd','wnbd','wrfd','q','r','ri','elo','tri','zeff','angf', 'f']
        td_ddas=["jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp","jsp" ,'jsp']
        td_tags=['temp_e','temp_i','dens_e','dens_h','dens_d','dens_t','dens_imp1','dens_imp2','dens_imp3','dens_imptot','z_imp1','z_imp2','z_imp3','dens_beam','dens_rf',
                'endens_beam','endens_rf','safety','majrad_out','majrad_in','elo','tri','zeff','tor_angv','polcurfunc' ]
        td_syms=[r'$T$',r'$T$',r'$n$',r'$n$',r'$n$',r'$n$',r'$n$',r'$n$',r'$n$',r'$n_{{imptot}}$',r'$Z_{{imp1}}$','$Z_{{imp2}}$','$Z_{{imp3}}$',r'$n_{{\rm{{beam}}}}$',r'$n_{{\rm{{RF}}}}$',
                r'$W_{{\rm{{beam}}}}$',r'$W_{{\rm{{RF}}}}$','$q$',r'$R_{{\rm{{out}}}}$',r'$R_{{\rm{{in}}}}$',r'$\kappa$',r'$\delta$',r'$Z_{{\rm{{eff}}}}$',r'$\omega_{{\rm{{tor}}}}$','$I$' ]

        for i in range(len(td_signals)):
            sigString = defaultStem + '/{}/{}{}'.format(td_ddas[i], td_signals[i], seq_post)
            try:
                node = sal.get(sigString)
            except:
                print('! --- {} not found at {} --- !'.format(td_tags[i], sigString))
                continue
            if not np.all(node.data==0.0):
                # Only store data if it contained some non-zero elements.
                twod[td_tags[i]] = tt_structs.twod_item(node, node.data, td_tags[i], zsym=td_syms[i])
                print('Found {} at {}.'.format(td_tags[i], sigString))
            else:
                print('! --- Array of zeros found for {} at {} --- !'.format(td_tags[i], sigString))
                 
        # Get zero-dimensional variables. Note that some of these are not actually zero-d! Anything with jst is actually time-dependent. 
        # This needs to be kept in mind when processing this data object.
        zd_signals=['btor','cur','nel','pnb','prf','tiax','teax','zeff','nimp','nish','zim1','zim2','zim3','aim1','aim2','aim3','zfd','afd']
        zd_ddas =  ["jst" ,"jst","jst","jst","jst","jst" ,"jst" ,"jst" ,"jss" ,"jss" ,"jss" ,"jss" ,"jss" ,"jss" ,"jss" ,"jss" ,"jss","jss"]
        zd_tags=['b_tor','plasma_cur','line_av_dens_e','p_nbi','p_rf','ax_temp_i','ax_temp_e','z_eff','num_imps','num_hyd_spec','z_imp1','z_imp2','z_imp3','mass_imp1','mass_imp2','mass_imp3','z_fast','mass_fast']
        zd_syms=['$B_T$','$I_P$','$<n_e>$',r'$P_{{\rm{{NBI}}}}$',r'$P_{{\rm{{RF}}}}$',r'$T_{{\rm{{ax}},i}}$',r'$T_{{\rm{{ax}},e}}$',r'$Z_{{\rm{{eff}}}}$',r'$N_{{\rm{{imps}}}}$',r'$N_{{\rm{{hyds}}}}$',
                r'$maxZimp1$','$maxZimp2$','$maxZimp3$','$massimp1$','$massimp2$','$massimp3$','$maxZfast$','$massfast$']
        
        for i in range(len(zd_signals)):
            sigString = defaultStem + '/{}/{}{}'.format(zd_ddas[i], zd_signals[i], seq_post)
            try:
                node = sal.get(sigString)
                print('Found {} at {}.'.format(zd_tags[i], sigString))
                if zd_ddas[i] == 'jst':
                    # If dda is jst, then the data comes as a one-d array against time. Pick the zero-d value at the time-index that most closely matches our time-choice.
                    itime = int((np.abs(node.dimensions[0].data-t_choice)).argmin())
                    data = tt_structs.oned_item(node, node.data, zd_tags[i]).reduce_dim(t_choice, interpolate).data
                    #data = node.data[itime]
                else:
                    data = node.data
                zerod[zd_tags[i]] = tt_structs.zerod_item(node, data, zd_tags[i], sym=zd_syms[i])
            except:
                print('! --- {} not found at {} --- !'.format(zd_tags[i], sigString))
                continue
 
        # Make entries for the masses and charges of the hydrogenic ions and electrons.
        for ispec in range(4):
            specLabel = ['h', 'd', 't', 'e'][ispec]
            if 'dens_{}'.format(specLabel) in twod:
                m = 'mass_{}'.format(specLabel)
                z = 'z_{}'.format(specLabel)
                zerod[m] = tt_structs.zerod_item( cp(zerod['b_tor'].signal), np.array([[ 1.0 , 2.0 , 3.0 , 5.4858e-4][ispec]]), m, description = 'Mass for species {}'.format(specLabel), units = '[amu]', sym='$M$')
                zerod[z] = tt_structs.zerod_item( cp(zerod['b_tor'].signal), np.array([[ 1.0 , 1.0 , 1.0 , -1.0 ][ispec]]), z, description = 'Atomic number for species {}'.format(specLabel), units = '', sym='$Z$')
        
        # Use Wesson's formula for electron-ion collisionality. Coulomb logarithm is loglam.
        loglam=14.9-0.5*np.log(1.0e-20*twod['dens_e'].data)+np.log(1.0e-3*twod['temp_e'].data)
        nu_ei = 917.4*(1.0e-19*twod['dens_e'].data)*loglam/np.power(1.0e-3*twod['temp_e'].data,1.5)

        ion_tags = ['h','d','t','imp1','imp2','imp3']
        Tion = twod['temp_i'].data
        Te = twod['temp_e'].data
        ne = twod['dens_e'].data
        me = 5.4858e-4                                  # In atomic mass units.
        for i in range(6):
            speci = ion_tags[i]
            densi = 'dens_{}'.format(speci)
            if densi in twod:
                mi = zerod['mass_{}'.format(speci)].data
                ni = twod[densi].data
                zi = zerod['z_{}'.format(speci)].data
                col_fac_sum = 0
                for j in range(6):
                    specj = ion_tags[j]
                    densj = 'dens_{}'.format(specj)
                    if densj in twod:
                        mj = zerod['mass_{}'.format(specj)].data
                        nj = twod[densj].data
                        zj = zerod['z_{}'.format(specj)].data
                        col_fac_sum = col_fac_sum + ( nj * zj**2 / ne ) * 2.0 / (1 + np.sqrt( mi/mj ))
                twod['nu_{}'.format(speci)] = tt_structs.twod_item( cp(twod['majrad_out'].signal), nu_ei * np.sqrt(me/mi) * np.power( Te/Tion , 1.5 ) * zi**2, 'nu_{}'.format(speci), zdescription = 'Collisionality or species {}'.format(speci), zunits = '[s-1]', zsym = r'$\nu$')
        twod['nu_e'] = tt_structs.twod_item( cp(twod['majrad_out'].signal), nu_ei, 'nu_e', zdescription = 'Collisionality of species e', zunits = '[s-1]', zsym = r'\nu_e' )    

        # Calculate minor and major radii.
        twod['minrad'] = tt_structs.twod_item(cp(twod['majrad_out'].signal), (cp(twod['majrad_out'].data) - cp(twod['majrad_in'].data))/2.0, 'minrad', zdescription = 'Minor radius', zsym = '$r$')
        twod['majrad'] = tt_structs.twod_item(cp(twod['majrad_out'].signal), (cp(twod['majrad_out'].data) + cp(twod['majrad_in'].data))/2.0, 'majrad', zdescription = 'Major radius', zsym = '$R_0$')

        # Calculate Rgeo. I have used the definition that I think is correct, which is different to CMR's script that uses majrad[:,-1]. 
        # I think it should be the poloidal current function on the flux surface of interest (i.e. a function of minrad), divided by the normalizing B field.
        twod['rgeo'] = tt_structs.twod_item( cp(twod['majrad'].signal), cp(twod['polcurfunc'].data)/cp(zerod['b_tor'].data), 'rgeo', zdescription = r'$R_{\rm{geo}}$', zsym = r'R_{\rm{geo}}')
        
        # Write down electron mass.
        
        return (zerod, oned, twod)
        
    def choose_time(self, t_choice, interpolate):
        
        # This function converts reduces the dimension of variables with a time dimension by picking out the time that we are interested in.
        # Could either pick the data values at the closest time, or (linearly) interpolate all values to get a hopefully better estimate of their value in the time of interest.
        
        # Do nothing for zero-d variables.

        # For one-d variables, check the units of the x-dimension. If it is time, then reduce dimensionality, remove from oned and add to zerod.
        for key in list(self.oned):
            item = self.oned[key]
            if item.xunits.lower() not in ['secs']:
                print("! --- x units are {}, not secs --- !".format(item.xunits.lower()))
                continue
            else:
                self.zerod[item.tag] = item.reduce_dim(t_choice, interpolate)
                del self.oned[item.tag]
                   
        # Do the same for two-d variables.
        for key in list(self.twod):
            item = self.twod[key]
            if item.xunits.lower() not in ['secs']:
                print("! --- x units are {}, not secs --- !".format(item.xunits.lower()))
                continue
            else:
                self.oned[item.tag] = item.reduce_dim(t_choice, interpolate)
                del self.twod[item.tag]

    # r choice is the normalized radius choice.
    def choose_radius(self, r_choice, interpolate):
        self.r_choice = r_choice

        # Now find the x-dimension value that corresponds to r_choice.
        xDat = self.oned['minrad'].signal.dimensions[0].data
        rDat = self.oned['minrad'].data
        i = int((np.abs(rDat-r_choice)).argmin())
        if rDat[i] < r_choice and i + 1 < len(xDat):
            i1 = i
            i2 = i+1
        elif rDat[i] > r_choice and i != 0:
            i1 = i-1
            i2 = i
        m = (xDat[i2]-xDat[i1]) / (rDat[i2]-rDat[i1])
        c = xDat[i1] - m*rDat[i1]
        x_choice = m*r_choice + c
        
        for key in list(self.oned):
            item = self.oned[key]
            self.zerod[item.tag] = item.reduce_dim(x_choice, interpolate)
            del self.oned[item.tag]

    def get_gradients(self):
        # First, we get the rate of change the arbitrary x-vector with rminor: dx/dr. 
        # All subsequent gradients can then be calculated as dF/dr = dF/dx * dx/dr
        dxdr = np.gradient(self.oned['minrad'].xdata,self.oned['minrad'].data)

        # Shafranov shift is dRmaj/dx * dx/dr
        self.oned['shift'] = tt_structs.oned_item(cp(self.oned['majrad'].signal), dxdr * np.gradient(self.oned['majrad'].data, self.oned['majrad'].xdata), 'shift', ydescription='Shafranov shift (R0prime)', yunits = '', ysym = r'$R^\prime_0$')
        
        # Magnetic shear is (r/q)*dq/dx*dx/dr
        self.oned['shat'] = tt_structs.oned_item(cp(self.oned['safety'].signal), self.oned['minrad'].data*dxdr*np.gradient(self.oned['safety'].data, self.oned['safety'].xdata)/self.oned['safety'].data, 'shat', ydescription='Magnetic shear', yunits = '',ysym = r'$\hat{{s}}$')
        
        # Elongation gradient is d(elo)/dx * dx/dr
        self.oned['delodr'] = tt_structs.oned_item(cp(self.oned['elo'].signal), dxdr * np.gradient(self.oned['elo'].data, self.oned['elo'].xdata), 'delodr', ydescription='Radial gradient of elongation', yunits = '[m-1]', ysym=r'$\kappa^\prime$')
        
        # Triangularity gradient is d(tri)/dx * dx/dr
        self.oned['dtridr'] = tt_structs.oned_item(cp(self.oned['tri'].signal), dxdr * np.gradient(self.oned['tri'].data, self.oned['tri'].xdata), 'dtridr', ydescription='Radial gradient of triangularity', yunits = '[m-1]', ysym = r'$\delta^\prime$')

        # Get ion and electron temperature gradients: tprim_S = -(1/T_S) d(T_S)/dx * dx/dr. All ions will be assumed to have the same temperature and temperature gradient, so don't bother making tprim_h,d,t,impX items.
        self.oned['tprim_i'] = tt_structs.oned_item(cp(self.oned['temp_i'].signal), -1.0*dxdr*np.gradient(self.oned['temp_i'].data, self.oned['temp_i'].xdata)/self.oned['temp_i'].data, 'tprim_i', 
            ydescription='1 / ( ion temperature length scale )', yunits = '[m-1]', ysym = r'$L_T^{{-1}}$')
        self.oned['tprim_e'] = tt_structs.oned_item(cp(self.oned['temp_e'].signal), -1.0*dxdr*np.gradient(self.oned['temp_e'].data, self.oned['temp_e'].xdata)/self.oned['temp_e'].data, 'tprim_e', 
            ydescription='1 / ( electron temperature length scale )', yunits = '[m-1]', ysym = r'$L_T^{{-1}}$')

        # Get flow shear rate.
        self.oned['g_exb'] = tt_structs.oned_item(cp(self.oned['tor_angv'].signal), dxdr*self.oned['minrad'].data*np.gradient(self.oned['tor_angv'].data, self.oned['tor_angv'].xdata)/self.oned['safety'].data, 'g_exb', 
            ydescription = 'ExB Flow shear rate', ysym = r'$\gamma_{{ExB}}$')

        # Total pressure in Pascals, and its radial derivative. These will be used down the line to calculate beta-prime and beta.
        total_p = np.zeros(len(self.oned['dens_e'].data))
        total_dpdr = np.zeros(len(total_p))

        # For each species:
        for spec in ['e', 'h', 'd', 't', 'imp1', 'imp2', 'imp3']:
            dens = 'dens_{}'.format(spec)
            fprim = 'fprim_{}'.format(spec)
            temp = 'temp_{}'.format(spec)
            tprim = 'tprim_{}'.format(spec)
            # Get species density gradients for recorded species: fprim_S = -(1/n_S) d(n_S)/dx * dx/dr. 
            if dens in self.oned:
                self.oned[fprim] = tt_structs.oned_item(cp(self.oned[dens].signal), -1.0 * dxdr * np.gradient(self.oned[dens].data, self.oned[dens].xdata) / self.oned[dens].data, fprim, 
                    ydescription='1 / ( {} density length scale )'.format(spec), yunits = '[m-1]', ysym = r'$L_n^{{-1}}$')

            if dens in self.oned and temp in self.oned:
                # This should just be the electrons.
                total_p = total_p + 1.6022e-19 * self.oned[dens].data * self.oned[temp].data
                total_dpdr = total_dpdr - 1.6022e-19 * self.oned[dens].data * self.oned[temp].data * ( self.oned[fprim].data + self.oned[tprim].data )
            elif dens in self.oned:
                # This should be the other ions. Note that we don't include a fast-particle contribution, but this should be doable using the signals given.
                total_p = total_p + 1.6022e-19 * self.oned[dens].data * self.oned['temp_i'].data
                total_dpdr = total_dpdr - 1.6022e-19 * self.oned[dens].data * self.oned['temp_i'].data * ( self.oned[fprim].data + self.oned['tprim_i'].data )

        # Tot[al pressure
        self.oned['p_tot'] = tt_structs.oned_item(cp(self.oned['temp_i'].signal), total_p, 'p_tot', ydescription='Total pressure', yunits='[Pa]', ysym = r'$P_{{\rm{{tot}}}}$')
        self.oned['dpdr_tot'] = tt_structs.oned_item(cp(self.oned['temp_i'].signal), total_dpdr, 'dpdr_tot', ydescription='r-derivative of total pressure', yunits='[Pa m-1]', ysym = r'$P^\prime_{{\rm{{tot}}}}$')


    def get_species_labels(self):
        # Matches atomic number to an element label.
        imp_dict = {2:'He', 3:'Li', 4:'Be', 5:'B', 6:'C'}
    
        self.spec_labels = {'e':'e'}
        for i in range(2):
            self.spec_labels[['h','d','t'][i]] = '${}_1^{}$'.format(['H','D','T'][i], i+1)

        for spec in ['imp1','imp2','imp3']:
            if 'mass_{}'.format(spec) in self.zerod:
                self.spec_labels[spec] = '${}^{{{}}}_{{{}}}$'.format(imp_dict[self.zerod['z_{}'.format(spec)].data[0]], int(self.zerod['mass_{}'.format(spec)].data[0]), int(self.zerod['z_{}'.format(spec)].data[0]))

    def normalize(self):
        # Normalizing length scale is the minor radius at the LCFS.
        Lref = self.oned['minrad'].data[-1]
        self.zerod['amin'] = tt_structs.zerod_item( cp(self.oned['minrad'].signal), [Lref], 'amin', description = 'Normalizing length scale (minor radius)', sym=r'$a_{\rm{min}}$')
    
        # Normalizing magnetic field is B_tor.
        Bref = self.zerod['b_tor'].data[0]

        # Normalizing mass is that of the lightest hydrogenic ion (in atomic mass units).
        for ispec in range(2):
            if 'dens_{}'.format(['h','d','t'][ispec]) in self.oned:
                mref = ispec + 1
                break
        self.zerod['m_ref'] = tt_structs.zerod_item(cp(self.zerod['amin'].signal), [mref], 'm_ref', description='Normalizing mass', units ='[amu]', sym = r'$M_{\rm{ref}}$')

        # ( Normalizing charge is 1 )

        # Normalizing temperature is ion temperature at r=0.
        Tref = self.oned['temp_i'].data[0]
        self.zerod['T_ref'] = tt_structs.zerod_item(cp(self.zerod['amin'].signal), [Tref], 'T_ref', description='Normalizing temperature', units ='[eV]', sym = r'$T_{\rm{ref}}$')
        
        # Normalizing velocity is sqrt(normalizing temperature, divided by normalizing mass).
        vref = np.sqrt(1.6022e-19*Tref/(2.0*mref*1.661e-27))
        self.zerod['v_ref'] = tt_structs.zerod_item(cp(self.zerod['amin'].signal), [vref], 'v_ref', description='Normalizing velocity', units ='[ms-1]', sym = r'$v_{\rm{ref}}$')

        # Normalizing density is electron density at r=0.
        nref = self.oned['dens_e'].data[0]
        self.zerod['n_ref'] = tt_structs.zerod_item(cp(self.zerod['amin'].signal), [nref], 'n_ref', description='Normalizing density', units ='[m-3]', sym = r'$n_{\rm{ref}}$')

        # Reference beta is reference density*reference temperature/reference magnetic field.
        betaRef = nref*Tref*4.0268e-25/(Bref*Bref)
        self.zerod['beta_ref'] = tt_structs.zerod_item(cp(self.zerod['amin'].signal), [betaRef], 'beta_ref', description='Normalizing beta', units ='', sym = r'$\beta_{\rm{ref}}$')

        # Normalize length scales like minrad, majrad, rgeo.
        self.oned['minrad'].normalize(Lref, r'$\frac{r}{ a_{\rm{min}} }$')
        self.oned['majrad'].normalize(Lref, r'$\frac{R_0}{ a_{\rm{min}} }$')
        self.oned['rgeo'].normalize(Lref, r'$\frac{R_{\rm{geo}}}{ a_{\rm{min}} }$')

        # Normalize gradients like delodr, dtridr.
        self.oned['delodr'].normalize(1.0/Lref, r'$a_{\rm{min}} \kappa^\prime $')
        self.oned['dtridr'].normalize(1.0/Lref, r'$a_{\rm{min}} \delta^\prime$')
        
        # Normalize species parameters.
        self.oned['temp_i'].normalize(Tref, r'$\frac{ T }{ T_{\rm{ref}} }$')
        self.oned['temp_e'].normalize(Tref, r'$\frac{ T }{ T_{\rm{ref}} }$')
        self.oned['tprim_i'].normalize(1.0/Lref, r'$\frac{ a_{\rm{min}} }{ L_T }$')
        self.oned['tprim_e'].normalize(1.0/Lref, r'$\frac{ a_{\rm{min}} }{ L_T }$')
        for spec in ['h', 'd', 't','e', 'imp1', 'imp2', 'imp3']:
            dens = 'dens_{}'.format(spec)
            if dens in self.oned:
                self.oned[dens].normalize(nref, r'$\frac{n}{n_{\rm{ref}}}$')
                self.oned['fprim_{}'.format(spec)].normalize(1.0/Lref, r'$\frac{ a_{\rm{min}} }{ L_n }$')
                self.zerod['mass_{}'.format(spec)].normalize(mref, r'$\frac{ M }{ a_{\rm{min}} }$')
                self.oned['nu_{}'.format(spec)].normalize(vref/Lref, r'$\nu \frac{ a_{\rm{min}} }{ v_{\rm{ref}} }$')
        self.oned['p_tot'].normalize(Bref*Bref/(2.5133e-6))
        self.oned['dpdr_tot'].normalize(Bref*Bref/(2.5133e-6))
        
        # Flow-related frequencies normalized to reference frequency.
        self.oned['g_exb'].normalize(vref/Lref, r'$\frac{ a_{\rm{min}} }{ v_{\rm{ref}} } \gamma_{ E\times B }$')
        # This is the mach number.
        self.oned['tor_angv'].normalize(vref/Lref, r'$\frac{ a_{\rm{min}} }{ v_{\rm{ref}} } v_{\rm{tor}}$')

    # Here, rho minor radius we are querying.
    def plot_profiles(self, rho):
        # Profiles to plot:
        #  - Temperature, Density, Tprim, fprim for ions, electrons.
        #  - Safety factor and Magnetic shear.
        #  - Elongation and triangularity, and their gradients.
        #  - Major radius, and its gradient.
        #  - Toroidal angular velocity and flow shear.        

        # Get formatted species labels.
        self.get_species_labels()

        pdflist = []
        merged_pdfname = '{}_{}_{}_{}_profiles.pdf'.format(self.shot, self.uid, self.seq, self.t_choice)
        tmp_pdf_id = 1
        
        xlab = '{} {}'.format(self.oned['minrad'].sym, self.oned['minrad'].units)
        
        # Species density and temperature.
        for spec in ['h', 'd', 't','e', 'imp1', 'imp2', 'imp3']:
            if spec == 'e':
                temp = 'temp_e'
                tprim = 'tprim_e'
            else:
                temp = 'temp_i'
                tprim = 'tprim_i'
            dens = 'dens_{}'.format(spec)
            fprim = 'fprim_{}'.format(spec)
            if dens in self.oned:
                fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(16,12), sharex=True)
                fig.suptitle('Species parameters for {}'.format(self.spec_labels[spec]))
                tt_plot.plot_1d(self.oned['minrad'].data, self.oned[dens].data, '', axes = axes[0][0], ylab = '{} {}'.format(self.oned[dens].sym, self.oned[dens].units), markxpos=rho)
                tt_plot.plot_1d(self.oned['minrad'].data, self.oned[temp].data, '', axes = axes[0][1], ylab = '{} {}'.format(self.oned[temp].sym, self.oned[temp].units), markxpos=rho)
                tt_plot.plot_1d(self.oned['minrad'].data, self.oned[fprim].data, xlab, axes = axes[1][0], ylab = '{} {}'.format(self.oned[fprim].sym, self.oned[fprim].units), markxpos=rho)
                axes[1][0].set_ylim(min(0, 1.2*np.amin(self.oned[fprim].data)), min(np.max(self.oned[fprim].data), 12.5))
                tt_plot.plot_1d(self.oned['minrad'].data, self.oned[tprim].data, xlab, axes = axes[1][1], ylab = '{} {}'.format(self.oned[tprim].sym, self.oned[tprim].units), markxpos=rho)
                axes[1][1].set_ylim(min(0, 1.2*np.amin(self.oned[tprim].data)), min(np.max(self.oned[tprim].data), 12.5))

                tmp_pdfname = 'tmp{}.pdf'.format(tmp_pdf_id)
                plt.savefig(tmp_pdfname)
                pdflist.append(tmp_pdfname)
                tmp_pdf_id = tmp_pdf_id+1
 
        # Safety factor and Magnetic shear.
        fig, axes = plt.subplots(ncols = 2, figsize=(16,8))
        fig.suptitle('Safety factor and magnetic shear')
        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['safety'].data, xlab, axes = axes[0], ylab = '{}'.format(self.oned['safety'].sym), markxpos=rho)
        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['shat'].data, xlab, axes = axes[1], ylab = '{}'.format(self.oned['shat'].sym), markxpos=rho)
        axes[0].set_ylim(bottom=0.0, top = None)

        tmp_pdfname = 'tmp{}.pdf'.format(tmp_pdf_id)
        plt.savefig(tmp_pdfname)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        # Elongation and triangularity profiles. 
        fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(16,12), sharex=True)
        fig.suptitle('Elongation and Triangularity')

        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['elo'].data, '', axes = axes[0][0], ylab = '{}'.format(self.oned['elo'].sym), markxpos=rho)
        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['tri'].data, '', axes = axes[0][1], ylab = '{}'.format(self.oned['tri'].sym), markxpos=rho)
        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['delodr'].data, xlab, axes = axes[1][0], ylab = '{} {}'.format(self.oned['delodr'].sym, self.oned['delodr'].units), markxpos=rho)
        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['dtridr'].data, xlab, axes = axes[1][1], ylab = '{} {}'.format(self.oned['dtridr'].sym, self.oned['dtridr'].units), markxpos=rho)

        tmp_pdfname = 'tmp{}.pdf'.format(tmp_pdf_id)
        plt.savefig(tmp_pdfname)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        
        # Major radius, Shafranov Shift and Pressure. 
        fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(16,12), sharex=True)
        fig.suptitle('Major radius and Pressure')
        
        if self.oned['p_tot'].units != "":
            ptotylab = r'$P_{{\rm{{tot}}}}$ {}'.format(self.oned['p_tot'].units)
            dpdrtotylab = r'$P^\prime{{}}_{{\rm{{tot}}}}$ {}'.format(self.oned['dpdr_tot'].units)
        else:
            ptotylab = r'$\beta_{\rm{tot}}$'
            dpdrtotylab = r'$\beta^\prime{}_{\rm{tot}}$'
            axes[0][1].axhline(self.zerod['beta_ref'].data[0],linestyle='--', color='red')
            axes[0][1].text(0.92, self.zerod['beta_ref'].data[0]+0.001, r'$\beta_{\rm{ref}}$', color='red')

        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['majrad'].data, '', axes = axes[0][0], ylab = '{} {}'.format(self.oned['majrad'].sym, self.oned['majrad'].units), markxpos=rho)
        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['p_tot'].data, '', axes = axes[0][1], ylab = ptotylab, markxpos=rho)
        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['shift'].data, xlab, axes = axes[1][0], ylab = '{}'.format(self.oned['shift'].sym), markxpos=rho)
        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['dpdr_tot'].data, xlab, axes = axes[1][1], ylab = dpdrtotylab, markxpos=rho)

        tmp_pdfname = 'tmp{}.pdf'.format(tmp_pdf_id)
        plt.savefig(tmp_pdfname)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
       
        # Toroidal angular velocity.
        fig, axes = plt.subplots(ncols = 2, figsize=(16,8))
        fig.suptitle('Toroidal rotation and flow shear')
        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['tor_angv'].data, xlab, axes = axes[0], ylab = r'{} {}'.format(self.oned['tor_angv'].sym, self.oned['tor_angv'].units), markxpos=rho)
        tt_plot.plot_1d(self.oned['minrad'].data, self.oned['g_exb'].data, xlab, axes = axes[1], ylab = r'{} {}'.format(self.oned['g_exb'].sym, self.oned['g_exb'].units), markxpos=rho)
        axes[0].set_ylim(bottom=0.0, top = None)

        tmp_pdfname = 'tmp{}.pdf'.format(tmp_pdf_id)
        plt.savefig(tmp_pdfname)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        plt.show()
        
        tt_plot.merge_pdfs(pdflist, merged_pdfname)

    def generate_input(self, input_template):
        with open(input_template) as nml_file:
            
            namelist = f90nml.read(nml_file)
            namelist.float_format = "1.2E"
             
            patch_namelist = f90nml.namelist.Namelist()
            patch_namelist['theta_grid_knobs'] = {
            'equilibrium_option': 'eik'
            }
            
            # Decimal places.
            dp = 3
            # Scientific notation format string
            fs = '{{:1.{}E}}'.format(dp)
            patch_namelist['theta_grid_parameters'] = {
            'rhoc'   :float(fs.format(round(self.zerod['minrad'].data[0],dp))),
            'rmaj'   :float(fs.format(round(self.zerod['majrad'].data[0],dp))),
            'r_geo'  :float(fs.format(round(self.zerod['rgeo'].data[0],dp))),
            'qinp'   :float(fs.format(round(self.zerod['safety'].data[0],dp))),
            'shat'   :float(fs.format(round(self.zerod['shat'].data[0],dp))),
            'shift'  :float(fs.format(round(self.zerod['shift'].data[0],dp))),
            'akappa' :float(fs.format(round(self.zerod['elo'].data[0],dp))),
            'akappri':float(fs.format(round(self.zerod['delodr'].data[0],dp))),
            'tri'    :float(fs.format(round(self.zerod['tri'].data[0],dp))),
            'tripri' :float(fs.format(round(self.zerod['dtridr'].data[0],dp)))
            }

            patch_namelist['parameters'] = {
            'zeff' :float(fs.format(round(self.zerod['zeff'].data[0],dp))),
            'beta' :float(fs.format(self.zerod['beta_ref'].data[0]))
            }
         
            patch_namelist['theta_grid_eik_knobs'] = {
            'iflux'           :0,
            'irho'            :2,
            'bishop'          :4,
            's_hat_input' :float(fs.format(round(self.zerod['shat'].data[0],dp))),
            'beta_prime_input' :float(fs.format(round(self.zerod['dpdr_tot'].data[0],dp)))
            }
           
            patch_namelist['species_knobs'] = {
            'nspec': int(1 + self.zerod['num_hyd_spec'].data[0] + self.zerod['num_imps'].data[0])
            }

            species_counter = 1
            for spec in ['h','d','t','e','imp1','imp2','imp3']:
                dens = 'dens_{}'.format(spec)
                if 'dens_{}'.format(spec) in self.zerod:
                    if spec == 'e':
                        specType = 'electron'
                        tprim = 'tprim_e'
                        temp = 'temp_e'
                    else:
                        specType = 'ion'
                        tprim = 'tprim_i'
                        temp = 'temp_i'
                    patch_namelist['species_parameters_{}'.format(species_counter)] = {
                    'z'     :float(fs.format(round(self.zerod['z_{}'.format(spec)].data[0],dp))),
                    'mass'  :float(fs.format(round(self.zerod['mass_{}'.format(spec)].data[0],dp))),
                    'dens'  :float(fs.format(round(self.zerod['dens_{}'.format(spec)].data[0],dp))),
                    'fprim' :float(fs.format(round(self.zerod['fprim_{}'.format(spec)].data[0],dp))),
                    'temp'  :float(fs.format(round(self.zerod[temp].data[0],dp))),
                    'tprim' :float(fs.format(round(self.zerod[tprim].data[0],dp))),
                    'vnewk' :float(fs.format(self.zerod['nu_{}'.format(spec)].data[0],dp)),
                    'type'  :specType
                    }
                    species_counter = species_counter + 1
            
            patch_namelist['species_knobs'] = {
            'g_exb' :float(fs.format(round(self.zerod['g_exb'].data[0],dp))),
            'mach' :float(fs.format(round(self.zerod['tor_angv'].data[0],dp)))
            }                    


            patch_namelist.float_format = "1.2E"
            # Save patch.
            newInputFile = '{}_{}_{}_{}_{}.in'.format(self.shot, self.uid, self.seq, self.t_choice, self.r_choice)
            f90nml.patch(input_template, patch_namelist, newInputFile)
            
            #namelist.patch(patch_namelist)
            #with open(newInputFile, 'w') as out_nml_file:
            #    nml.write(out_nml_file)

            print('Successfully patched input file: {}.'.format(newInputFile))

