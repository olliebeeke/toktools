# Main script for getting and plotting data from experiment.
import tt_surface
import pickle

#################################################################################################
# User parameters:

# Desired time. Script will automatically find the data with the nearest time to the desired time.
t_choice = 48.0

# Data source. Currently supports only jetto data.
source = 'jetto'

# Flux surface label
rho = 0.33

# Shot number
shot = 53521

# User ID
uid = 'mroma'

# Sequence number. If -1, then the most recent one will be chosen. 
seq = -1

# If True we should interpolate between the nearest data points in time/radius.
interpolateTChoice = True
interpolateRChoice = True

# If specified, we write the surface data to a file.
writeSurfFile = 'tt_surface.dat'

# If specified, we read the surface data from a file.
readSurfFile = writeSurfFile
#readSurfFile = ''

# If true, plots figures.
with_plot = True

# If true, plots radial profiles AFTER normalization. Otherwise plots them with units.
plot_normalized = True

# Template input file name.
input_template = '/home/beekel/scripts/toktools/example.in'

################################################################################################

if readSurfFile == '':
    # Read the profile data from the relevant source.
    surface = tt_surface.surfobj(source=source, shot=shot, uid = uid, seq=seq, t_choice=t_choice, interpolate = interpolateTChoice)
else:
    print("Reading surface data from local file {}...".format(readSurfFile))
    with open(readSurfFile, 'rb') as read_file:
        surface = pickle.load(read_file)

if writeSurfFile != '' and readSurfFile != writeSurfFile:
    print("Writing surface data to local file {}...".format(writeSurfFile))
    with open(writeSurfFile, 'wb') as write_file:
            pickle.dump(surface,write_file)

# Get profile data at the time of interest. True/False indicates whether we want to interpolate. (True interpolate, False nearest)
print('Selecting time-slice of interest...')
surface.choose_time(t_choice, interpolateTChoice)

print('Getting radial gradients...')
# Get radial gradients for relevant quantities.
surface.get_gradients()


# Plot non-normalized profiles
if not plot_normalized and with_plot:
    print('Plotting non-normalized profiles...')
    surface.plot_profiles(rho*surface.oned['minrad'].data[-1])

print('Normalizing variables...')
surface.normalize()

if plot_normalized and with_plot:
    print('Plotting normalized profiles...')
    surface.plot_profiles(rho)

surface.choose_radius(rho, interpolateRChoice)

print('Generating input file based on input file: {}.'.format(input_template))
surface.generate_input(input_template)
    
