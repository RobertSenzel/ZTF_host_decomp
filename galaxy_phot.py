import bs4
import lxml
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from extinction import fm07
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from astroquery.ipac.ned import Ned
from astroquery.vizier import Vizier
from astropy.cosmology import Planck18
from matplotlib.patches import Ellipse
from skimage.measure import EllipseModel
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from astroquery.exceptions import RemoteServiceError


class HostGal:
    def __init__(self, verbose):
        self.verbose = verbose
        self.cutout = {}
        self.iso = {} 
        self.gal = {}
        self.survey = 'None'
 

    def init_query(self, host_name, sn_name, df):
        self.host_name = host_name
        if sn_name[0] in ['1', '2']:
            sn_name = 'SN' + sn_name
        self.sn_name = sn_name
        self.gal = self.get_coords(host_name, sn_name)
        self.df_lilit = self.lilit(host_name, df)
    

    def init_dr2(self, sn_name, sample, host_data):
        self.host_name = 'DR2'
        self.sn_name = sn_name
        host_ra, host_dec = host_data[['host_ra', 'host_dec']].loc[sn_name]
        mwebv, mwr_v, sn_ra, sn_dec, z, z_source = sample.data[['mwebv', 'mwr_v', 'ra', 
                                                                'dec', 'redshift', 'source']].loc[sn_name]
        if host_dec == -80:
            host_ra, host_dec = sn_ra, sn_dec
            print('no host') if self.verbose else None
        print(sn_name, host_ra, host_dec, z, z_source) if self.verbose else None
        self.gal = {'host': [host_ra, host_dec], 'z': z, 'sn': [sn_ra, sn_dec],
                     'A': mwebv * mwr_v, 'z_source': z_source}
        

    def get_coords(self, host_name, sn_name):

        def NED(host_query):
            ra, dec =  host_query['RA'][0], host_query['DEC'][0]
            print(host_name, sn_name, ra, dec, host_query['Redshift'][0]) if self.verbose else None
            redshift = host_query['Redshift']
            redshift = [-1] if (len(redshift) == 0) or (type(redshift[0]) == np.ma.core.MaskedConstant) else redshift
            return {'host': [ra, dec], 'z': float(redshift[0]), 'sn': [sn_query['RA'][0], sn_query['DEC'][0]]}

        def Hyper_Leda():
            host_query = Vizier.query_object(host_name, catalog=['HyperLeda'])[0][0]
            loc = SkyCoord(host_query['RAJ2000'], host_query['DEJ2000'], unit=(u.hourangle, u.deg))
            ra, dec = loc.ra.deg, loc.dec.deg
            print(host_name, sn_name, ra, dec, 'HL') if self.verbose else None
            return {'host': [ra, dec], 'z': -1, 'sn': [sn_query['RA'][0], sn_query['DEC'][0]]}

        sn_query = Ned.query_object(sn_name)
        if len(sn_query['RA']) == 0:
            raise TypeError
        try:
            host_query = Ned.query_object(host_name)
        except RemoteServiceError:
            return Hyper_Leda()
        if (len(host_query['RA']) == 0) or (len(host_query['DEC']) == 0):
           return Hyper_Leda()
        else:
           return NED(host_query)
    

    @staticmethod
    def lilit(host_name, df):
        vals = df[df.Gal.apply(lambda x: x.strip())==host_name].values[0]
        Usn, Vsn, R25, Z25 = vals[[2, 3, 10, 11]].astype(float)
        return {'a': R25, 'b': Z25, 'Usn': Usn, 'Vsn': Vsn}
        

    def get_cutout(self, survey, output_size, band, scale=0.262):
        ra, dec = self.gal['host']

        def legacy_url(survey):
            layer = 'ls-dr10' if survey == 'legacy' else 'sdss'
            service = 'https://www.legacysurvey.org/viewer/'
            return f'{service}fits-cutout?ra={ra}&dec={dec}&layer={layer}&pixscale={scale}&bands={band}&size={output_size}'

        def ps1_url():
            size = round(output_size*scale/0.25)
            service = 'https://ps1images.stsci.edu/cgi-bin/'
            file_url = f'{service}ps1filenames.py?ra={ra}&dec={dec}&filters={band}'
            table = Table.read(file_url, format='ascii')
            if len(table['filename']) == 0:
                return 'no file'
            else:
                return f"{service}fitscut.cgi?ra={ra}&dec={dec}&size={size}&format=fits&output_size={output_size}&red={table['filename'][0]}"
        
        if survey == 'auto':
            url = legacy_url('legacy')
            res = requests.get(url, timeout=10)
            self.survey = 'legacy'
            if len(res.content) > 1000:
                return fits.open(BytesIO(res.content))
            else:
                url = ps1_url()
                self.survey = 'ps1'
                if url == 'no file':
                    url = legacy_url('sdss')
                    self.survey = 'sdss'
                res = requests.get(url, timeout=10)
                if len(res.content) > 1000:
                    return fits.open(BytesIO(res.content))
                else:
                    raise TypeError
        elif (survey == 'ps1'):
            url = ps1_url()
            res = requests.get(url)
            self.survey = 'ps1'
            return fits.open(BytesIO(res.content))
        else:
            url = legacy_url(survey)
            res = requests.get(url)
            self.survey = survey
            return fits.open(BytesIO(res.content))
    

    def calc_incl(self, t, r25):
        def get_r0(t):
            if t > 7:
                return 0.38
            elif (t >= -5)  and (t <= 7):
                return 0.43 + 0.053*t
            else:
                return 0
        
        r0 = 10**get_r0(t)
        sini2 = (1-r25**(-2))/(1-r0**(-2))
        incl = np.rad2deg(np.arcsin(np.sqrt(sini2)))
        return incl if ~np.isnan(incl) else -1
    

    def hyper_leda(self):
        ra, dec = self.gal['host']
        co = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        result_table = Vizier.query_region(co, radius=0.1 * u.arcmin, catalog='HyperLeda')
        if (len(result_table) == 0):
            self.gal['type'] = 'none'
            self.gal['t'] = [-9, 20]
            self.gal['incl'] = -1
            self.gal['source'] = 'Hyperleda'
        else:
            gal_name = result_table[0]['PGC'][0]
            url = f'https://leda.univ-lyon1.fr/ledacat.cgi?PGC{gal_name}&ob=ra'
            out = requests.get(url)
            soup = bs4.BeautifulSoup(out.content, 'lxml')
            table_source = soup.find_all('table')[5]
            table = pd.DataFrame(columns=['Parameter', 'Value', 'Unit', 'Description'])
            for j in table_source.find_all('tr')[1:]:
                row_data = j.find_all('td')
                row = [i.text for i in row_data]
                table.loc[len(table)] = row

            dict_ = dict(zip(table['Parameter'].values, table['Value'].values))
            tcode = dict_.get('t', '-9 \pm 20').split()
            t = float(tcode[0])
            t_err = float(tcode[-1]) if len(tcode) == 3 else 20

            self.gal['type'] = dict_.get('type', 'none').strip()
            self.gal['source'] = 'hyperleda'
            self.gal['t'] = [t, t_err]
            self.gal['incl'] = float(dict_.get('incl', '-1').strip())


    def get_galaxy_params(self, source='hl', args=[0, 0, 6]):
        def manual(t, gal_type, source):
            a, b, t = self.iso.get('a', [args[0]])[0], self.iso.get('b', [args[1]])[0], args[2]
            self.gal['type'] = gal_type
            self.gal['t'] = [t, 20]
            self.gal['incl'] = self.calc_incl(t, a/b)
            self.gal['source'] = source

        if source == 'hl':
            self.hyper_leda()
        elif source == 'calc':
            manual(t=args[2], gal_type='none', source='calculated')
        elif source == 'auto':
            self.hyper_leda()
            if self.gal['incl'] != -1:
                pass
            elif (self.gal['incl'] == -1) & (self.gal['t'][0] != -9):
                manual(t=self.gal['t'][0], gal_type=self.gal['type'], source='hl_calculated')
            else:
                manual(t=args[2], gal_type='none', source='calculated')

        print(self.gal['type'], self.gal['t'], self.gal['incl'], self.gal['source'] ) if self.verbose else None
    
    
    def flux2mag(self, flux, survey, band, scale, soft, exptime=1, extinction=True):
        # ztf_filters = {'g': 4746.48, 'r': 6366.38, 'i': 7829.03}
        ps1_filters = {'g': 4810.16 , 'r': 6155.47, 'i': 7503.03, 'z': 8668.36, 'y': 9613.60}
        sdss_filters = {'g': 4671.78, 'r': 6141.12, 'i': 7457.89, 'z': 8922.78, 'u': 3608.04}
        leg_filters = {'g': 4769.90, 'r': 6370.44 ,'i': 7774.30, 'z': 9154.88, 'y': 9886.45, 'u': 3856.88} # DECam
        eff_wl = {'legacy': leg_filters[band], 'ps1': ps1_filters[band], 'sdss': sdss_filters[band]}

        A_ = fm07(np.array([eff_wl[survey]]), self.gal['A'])

        def mag_soft(flux, zp):
            b = {'g': 0.9e-10, 'r': 1.2e-10, 'i': 1.8e-10, 'z': 7.4e-10}
            return zp + -2.5/np.log(10) * (np.arcsinh(flux/(2*b[band]*scale**2)) + np.log(b[band])) - A_
        
        def mag_(flux, zp):
            return zp - 2.5*np.log10(flux/scale**2) - A_
        
        zp_ = {'legacy': 22.5, 'sdss': 22.5, 'ps1': 25 + 2.5*np.log10(exptime)}
        return mag_soft(flux, zp_[survey]) if soft else mag_(flux, zp_[survey])

    
    @staticmethod
    def calc_aperature(size, redshift, scale):
        if size == 'z':
            if np.sign(redshift) == -1:
                return 1000
            else:
                aperature = 0.08 # estimate_radius()
                dist = Planck18.luminosity_distance(redshift)
                return min(int(np.rad2deg(aperature/dist.value)*(3600/scale)),  1000)
        else:
            return size


    def get_image(self, source, survey, output_, band, scale, soft=True):
        folder_name = 'fits' if survey == 'auto' else survey
        path = f'dr2_{folder_name}_{band}/{self.sn_name}.fits'
        if source == 'query':
            output_size = self.calc_aperature(output_, self.gal['z'], scale)
            fits_ = self.get_cutout(survey, output_size, band, scale)
            fits_data = fits_[0]
        elif source == 'save':
            fits_ = fits.open(path)
            fits_data = fits_[0]
            self.survey = fits_data.header['survey']
            output_size = fits_data.header['output_size']
        elif source == 'query_save':
            output_size = self.calc_aperature(output_, self.gal['z'], scale)
            fits_ = self.get_cutout(survey, output_size, band, scale)
            fits_data = fits_[0]
            fits_data.header['survey'] = self.survey
            fits_data.header['output_size'] = output_size
            fits_data.writeto(path, overwrite=True)
            fits_.close()
            return
        else:
            print('invalid source')
            return

        print(self.survey, output_size, scale) if self.verbose else None
        flux, wcs = fits_data.data, WCS(fits_data.header)
        exptime = fits_data.header['exptime'] if self.survey == 'ps1' else 1
        mag = self.flux2mag(flux, self.survey, band, scale, soft, exptime=exptime)
        self.cutout = {'flux': flux, 'mag': mag, 'wcs': wcs, 'scale': scale}
        fits_.close()


    def fit_ellipse(self, isophote):

        def prep(iso, blur):
            def chop_boundary(data, size, depth):
                cond1 = ~np.isin(data.T[0], list(range(depth)))
                cond2 = ~np.isin(data.T[1], list(range(depth)))
                cond3 = ~np.isin(data.T[0],list(range(size-depth, size)))
                cond4 = ~np.isin(data.T[1],list(range(size-depth, size)))
                return data[cond1 & cond2 & cond3 & cond4]

            mag = self.cutout['mag']
            kernal = np.array([ [ 0,  0,  1,  1,  1],
                                [ 0,  0,  0,  1,  1],
                                [-1,  0,  0,  0,  1],
                                [-1, -1,  0,  0,  0],
                                [-1, -1, -1,  0,  0]])

            kernal_a = 1/(kernal + iso)
            kernal_ = kernal_a / np.sum(kernal_a)
            conv_blur = gaussian_filter(mag, sigma=blur)
            conv1 = convolve2d(conv_blur, kernal_, mode='same')
            conv2 = convolve2d(conv1, kernal_[::-1], mode='same')

            for wi in np.arange(0.1, 0.5, 0.1): 
                contour = np.stack(np.where((conv2.T > iso-wi) & (conv2.T < iso+wi)))
                if len(contour.T) > 1000:
                    break
            
            contour_a = contour.T[::max(len(contour.T)//5000, 1)]
            contour_r = chop_boundary(contour_a, len(mag), 5)
            return contour_r

        def two_pass(data, connect):

            def pass_one(regions):
                current_region = 0
                for x, y in data:
                    neighbours = regions[x-connect:x, y-connect:y+connect-1]
                    valid_ = neighbours[np.nonzero(neighbours)]
                    if np.any(valid_):
                        root_region = np.min(valid_)
                        regions[x, y] = root_region
                        region_set = set(valid_)
                        for i in region_set:
                            if i != root_region:
                                equivalent_labels[i] = root_region
                    else:
                        current_region += 1
                        regions[x, y] = current_region
                return regions
            
            def pass_two(regions):
                for x, y in data:
                    curr_region = regions[x, y]
                    regions[x, y] = equivalent_labels.get(curr_region, curr_region)
                return regions
            
            def get_pixels(count_):
                return np.stack(np.where(regions == region_count[0][np.where(region_count[1] == count_)]))

            def region_loc(count_):
                region_i = get_pixels(count_)
                return np.sum((region_i.mean(axis=1) - len(regions)//2)**2)

            mag = self.cutout['mag']
            regions = np.zeros_like(mag, dtype=int)
            equivalent_labels = {}

            regions1 = pass_one(regions)
            regions = pass_two(regions1)
            
            region_count = np.asarray(np.unique(regions, return_counts=True))
            lim_ = min(len(region_count[1]), 7)
            largest_regions = np.partition(region_count[1], -np.arange(2, lim_))[-np.arange(2, lim_)]
            largest_regions = largest_regions[largest_regions > 50]
            if len(largest_regions) == 0:
                return []
            else:
                arg_ = np.argmin(np.vectorize(region_loc)(largest_regions))
                return get_pixels(largest_regions[arg_]).T
        
        contour_r = prep(isophote, blur=2)
        px_fit = two_pass(contour_r, connect=10)
        if len(px_fit) >= 30:
            ell = EllipseModel()
            ell.estimate(px_fit)
            res = ell.residuals(px_fit)
            return ell.params, (0,0,0,0,0), (contour_r, px_fit), np.sum(res**2)/len(res)
        else:
            return (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), ([], []), 1000


    def fit_ellipse_old(self, isophote):
        @numba.njit()
        def feature_removal(data, size, width, dim):
            for i in range(0, dim, width*2):
                for j in range(0, dim, width*2):
                    if (dim/2-i)**2 + (dim/2-j)**2 < (size*1.5)**2:
                        continue
                    vec = data - np.array([i, j])
                    dists = np.sqrt(vec.T[0]**2 + vec.T[1]**2)
                    px_center = np.sum(dists < size)
                    px_edge = np.sum((dists > size) & (dists <= size+width))
                    if (px_center > 0) and (px_edge == 0):
                        data = data[dists > size]
            return data
        
        def chop_boundary(data, size, depth):
            cond1 = ~np.isin(data.T[0], list(range(depth)))
            cond2 = ~np.isin(data.T[1], list(range(depth)))
            cond3 = ~np.isin(data.T[0],list(range(size-depth, size)))
            cond4 = ~np.isin(data.T[1],list(range(size-depth, size)))
            return data[cond1 & cond2 & cond3 & cond4]

        def run_fit(window):
            kernal = np.array([[ 0,  0,  1,  1,  1],
                               [ 0,  0,  0,  1,  1],
                               [-1,  0,  0,  0,  1],
                               [-1, -1,  0,  0,  0],
                               [-1, -1, -1,  0,  0]])
            
            kernal_a = 1/(kernal + isophote)
            kernal_ = kernal_a / np.sum(kernal_a)
            
            mag = self.cutout['mag']
            conv1 = convolve2d(mag, kernal_, mode='same')
            conv2 = convolve2d(conv1, kernal_[::-1], mode='same')
            contour = np.stack(np.where((conv2.T > isophote-window) & (conv2.T < isophote+window)))
            contour = contour.T[::max(len(contour.T)//3000, 1)]
            contour_r = chop_boundary(contour, len(mag), 5)
            
            noise = np.zeros(len(contour_r))
            for i in range(len(contour_r)):
                dists = np.linalg.norm(contour_r[i]-contour_r, axis=1)
                dists.sort()
                noise[i] = dists[1:8].mean()
            contour_r = contour_r[noise < 10]

            contour_small = feature_removal(contour_r, 50, 5, len(mag)) 
            contour_medium = feature_removal(contour_small, 150, 5, len(mag)) 
            
            px_fit = contour_medium.copy()
            if len(px_fit) >= 30:
                ell = EllipseModel()
                ell.estimate(px_fit)
                res = ell.residuals(px_fit)
                return ell.params, np.sum(res**2)/len(res), contour, contour_r, px_fit
            else:
                return (0, 0, 0, 0, 0), 1000, [], [], []

        monte_carlo_windows = np.linspace(0.1, 0.7, 10)
        data = np.zeros((len(monte_carlo_windows), 6))
        collect_points = []
        for i, wi in enumerate(monte_carlo_windows):
            (xc, yc, a, b, theta), chi2, all_points, reduced_points, fitted_points = run_fit(wi)
            collect_points.append([all_points, reduced_points, fitted_points])
            data[i] = (xc, yc, a, b, theta, chi2)

        data_r = data[data.T[0] != 0]
        if len(data_r) != 0:
            mean_vals = np.average(data_r.T, axis=1, weights=1/data_r.T[5])
            err_vals = np.std(data_r.T, axis=1)

            if np.std(data_r.T[5]) > np.mean(data_r.T[5]):
                converged_ = np.where(data_r.T[5] < data_r.T[5].mean())[0]
                collect_points = np.array(collect_points)[converged_] 

            ind_ = np.argmax(list(map(lambda x: len(x[2]), collect_points)))
            return mean_vals[:5], err_vals[:5], collect_points[ind_], data.T[5][ind_]
        else:
            return (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), [[], [], []], 1000
 

    def fit_isophote(self, isophote, plot=False):
        wcs = self.cutout['wcs']
        vals, errs, (px_all, px_fit), chi2 = self.fit_ellipse(isophote)
        (xc, yc, a, b, theta), (xc_err, yc_err, a_err, b_err, theta_err) =  vals, errs
        if a > 0:
            sn_ra, sn_dec = self.gal['sn']
            sn_loc = np.stack(wcs.world_to_pixel(SkyCoord(sn_ra, sn_dec, unit=u.deg)))
            rm = np.array([[np.cos(-theta), np.sin(-theta)], [-np.sin(-theta), np.cos(-theta)]]).T
            Usn, Vsn =  rm @ (sn_loc-[xc, yc])
        else: 
            Usn, Vsn = 0, 0
        self.iso = {'a': [a, a_err], 'b': [b, b_err], 'center': [(xc, yc), (xc_err, yc_err)], 
                    'angle': [theta, theta_err] , 'Usn': [Usn, xc_err], 'Vsn': [Vsn, yc_err], 'isophote': isophote,
                    'px_all': px_all,  'px_fit': px_fit, 'chi2': chi2}

        if plot and (len(px_all) > 0):
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            ax.scatter(px_all.T[0], px_all.T[1], color='b', s=3, marker='o')
            ax.scatter(px_fit.T[0], px_fit.T[1], color='lime', s=3, marker='o')
            ell_patch = Ellipse((xc, yc), 2*a, 2*b, np.rad2deg(theta), edgecolor='black', facecolor='none')
            ax.add_patch(ell_patch)
            plt.show()
                        

    def plot(self, isophote=False, check=False, save=[False, ''], px_depth=1):
        wcs, mag, scale = self.cutout['wcs'], self.cutout['mag'], self.cutout['scale']
        (sn_ra, sn_dec) = self.gal['sn']

        fig, ax = plt.subplots(figsize=(7, 6), dpi=100, subplot_kw={'projection': wcs})
        map_ = ax.imshow(mag, cmap='gray')
        c = SkyCoord(ra=sn_ra*u.degree, dec=sn_dec*u.degree, frame='icrs')
        sn_px = wcs.world_to_pixel(c)
        ax.scatter(sn_px[0], sn_px[1], c='cyan', s=40, lw=2, fc='None', ec='cyan')

        fig.colorbar(map_, ax=ax, label=r'mag arcsec$^{-2}$')
        ax.set_ylabel('Declination')
        ax.set_xlabel('Right Ascension')
        ax.set_title(f'{self.host_name}, {self.sn_name}')
        ax.coords[0].set_format_unit(u.deg, decimal=True)
        ax.coords[1].set_format_unit(u.deg, decimal=True)

        if isophote:
            a1, b1, theta, iso = self.iso['a'][0], self.iso['b'][0], self.iso['angle'][0], self.iso['isophote']
            (xc, yc), Usn, Vsn = self.iso['center'][0], self.iso['Usn'][0], self.iso['Vsn'][0]
            px_iso = np.stack(np.where(np.round(mag.T, px_depth) == iso))
            ax.scatter(px_iso[0], px_iso[1], color='r', s=3, ec='black', lw=0.1, marker='o')
            ell_patch1 = Ellipse((xc, yc), 2*a1, 2*b1, np.rad2deg(theta), edgecolor='lime', facecolor='none')
            ax.add_patch(ell_patch1)

            x_h = np.linspace(0, Usn, 10)
            y_h = np.zeros_like(x_h)
            y_v = np.linspace(0, Vsn, 10)
            x_v = np.ones_like(y_v)*Usn
            rm = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T
            x_hr, y_hr = rm @ np.stack([x_h, y_h])
            x_vr, y_vr = rm @ np.stack([x_v, y_v])

            ax.plot(x_hr+xc, y_hr+yc, 'b')
            ax.plot(x_vr+xc, y_vr+yc, 'b')
           
            if check:
                a2, b2, r25, z25 = self.df_lilit.values()
                ell_patch2 = Ellipse((xc, yc), 2*a2/scale, 2*b2/scale, np.rad2deg(theta), edgecolor='orange', facecolor='none')
                ax.add_patch(ell_patch2)

        if save[0]:
            fig.savefig(f'{save[1]}/image_{self.sn_name}.png')
            plt.close();
        else:
            return fig, ax
    

    def gal_scale(self, args=[0, 0, 0, 0]):
        dist = Planck18.luminosity_distance(self.gal['z'])
        a, b = self.iso.get('a', [args[0]])[0], self.iso.get('b', [args[1]])[0]
        Usn, Vsn =  self.iso.get('Usn', [args[2]])[0], self.iso.get('Vsn', [args[3]])[0]
        sc = self.cutout['scale']
        self.gal['a_kpc'] = np.deg2rad(a*sc/3600) * dist.to(u.kpc).value
        self.gal['b_kpc'] = np.deg2rad(b*sc/3600) * dist.to(u.kpc).value
        self.gal['sep'] = np.deg2rad(np.sqrt(Usn**2 + Vsn**2)*sc/3600) * dist.to(u.kpc).value
        print(self.gal['a_kpc'], self.gal['b_kpc'], self.gal['sep']) if self.verbose else None


    def sky_coverage(self, survey):
        if survey == 'legacy':
            df = Table.read('survey-bricks-dr10-south.fits.gz', format='fits')
            ra_l, ra_u, dec_l, dec_u = df[['ra1', 'ra2', 'dec1', 'dec2']].to_pandas().values.T
            ra, dec = df[['ra', 'dec']].to_pandas().values.T
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(ra, dec, 'b.', ms=0.3)
            ax.axhline(34, color='red')
            ax.set_ylabel('Declination')
            ax.set_xlabel('Right Ascension')
        elif survey == 'ps1':
            df = fits.open('ps1grid.fits')
