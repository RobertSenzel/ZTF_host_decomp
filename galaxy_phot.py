import bs4
import lxml
import ztfidr
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
from skimage import measure, draw
from scipy.signal import convolve2d
from astroquery.ipac.ned import Ned
from astroquery.vizier import Vizier
from astropy.cosmology import Planck18
from matplotlib.patches import Ellipse
from scipy.interpolate import interp1d
from skimage.measure import EllipseModel
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from scipy.special import gamma, gammainc
from scipy.optimize import curve_fit, fsolve
from skimage.measure import label as ConnectRegion
from astroquery.exceptions import RemoteServiceError
from scipy.odr import Model, Data, ODR

sample = ztfidr.get_sample()
host_data = ztfidr.io.get_host_data()
Vizier.ROW_LIMIT = 1000
qs = Vizier.get_catalogs('J/ApJS/196/11/table2')
df_update = pd.read_csv('csv_files/ztfdr2_matched_hosts.csv', index_col=0)

class HostGal:
    def __init__(self, verbose):
        self.verbose = verbose
        self.cutout = {}
        self.iso = {} 
        self.gal = {}
        self.survey = 'None'
 

    def init_query(self, host_name, catalog):
        self.host_name = host_name
        self.sn_name = 'None'

        if catalog == 'virgo':
            query = Vizier.query_object(host_name, catalog='J/AJ/90/1681')
            sc = SkyCoord(ra=query[0]['_RA.icrs'][0], dec=query[0]['_DE.icrs'][0],  unit=(u.hourangle, u.deg))
            self.gal = {'host': [sc.ra.deg, sc.dec.deg], 'z': 0.004, 'sn': [sc.ra.deg, sc.dec.deg],
                        'A': 0, 'z_source': 'None'}
        
        if catalog == 'sdss':
            self.gal = {'host': [qs[0][host_name]['_RA'], qs[0][host_name]['_DE']], 'z': qs[0][host_name]['z'], 
                        'sn': [qs[0][host_name]['_RA'], qs[0][host_name]['_DE']], 'A': 0, 'z_source': 'None'}
            print(self.gal['z']) if self.verbose else None


    def init_dr2(self, sn_name):
        self.host_name = 'DR2'
        self.sn_name = sn_name

        dz = (sample.data['redshift'] - df_update['z']).dropna()
        delta_z = dz[dz!=0]
        ztf_z = sample.data['redshift']
        ztf_z.loc[delta_z.index] = df_update['z'].loc[delta_z.index]
        z = ztf_z.loc[sn_name]

        host_ra, host_dec = df_update[['ra', 'dec']].loc[sn_name]
        if np.isnan(host_ra) or np.isnan(host_dec):
            host_ra, host_dec = host_data[['host_ra', 'host_dec']].loc[sn_name]
            
        mwebv, mwr_v, sn_ra, sn_dec, z_source = sample.data[['mwebv', 'mwr_v', 'ra', 
                                                                'dec', 'source']].loc[sn_name]
        if host_dec == -80:
            host_ra, host_dec = sn_ra, sn_dec
            print('no host') if self.verbose else None
        print(sn_name, host_ra, host_dec, z, z_source) if self.verbose else None
        self.gal = {'host': [host_ra, host_dec], 'z': z, 'sn': [sn_ra, sn_dec],
                     'A': mwebv * mwr_v, 'z_source': z_source}
        

    def get_coords(self, host_name, sn_name):
        pass
        # def NED(host_query):
        #     ra, dec =  host_query['RA'][0], host_query['DEC'][0]
        #     print(host_name, sn_name, ra, dec, host_query['Redshift'][0]) if self.verbose else None
        #     redshift = host_query['Redshift']
        #     redshift = [-1] if (len(redshift) == 0) or (type(redshift[0]) == np.ma.core.MaskedConstant) else redshift
        #     return {'host': [ra, dec], 'z': float(redshift[0]), 'sn': [sn_query['RA'][0], sn_query['DEC'][0]]}

        # def Hyper_Leda():
        #     host_query = Vizier.query_object(host_name, catalog=['HyperLeda'])[0][0]
        #     loc = SkyCoord(host_query['RAJ2000'], host_query['DEJ2000'], unit=(u.hourangle, u.deg))
        #     ra, dec = loc.ra.deg, loc.dec.deg
        #     print(host_name, sn_name, ra, dec, 'HL') if self.verbose else None
        #     return {'host': [ra, dec], 'z': -1, 'sn': [sn_query['RA'][0], sn_query['DEC'][0]]}

        # sn_query = Ned.query_object(sn_name)
        # if len(sn_query['RA']) == 0:
        #     raise TypeError
        # try:
        #     host_query = Ned.query_object(host_name)
        # except RemoteServiceError:
        #     return Hyper_Leda()
        # if (len(host_query['RA']) == 0) or (len(host_query['DEC']) == 0):
        #    return Hyper_Leda()
        # else:
        #    return NED(host_query)
   

    def get_cutout(self, survey, output_size, band, scale=0.262):
        ra, dec = self.gal['host']

        def legacy_url(survey):
            layer = 'ls-dr10' if survey == 'legacy' else 'sdss'
            service = 'https://www.legacysurvey.org/viewer/'
            return f'{service}fits-cutout?ra={ra}&dec={dec}&layer={layer}&pixscale={scale}&bands={band}&size={output_size}&subimage'

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
        

    def hyper_leda(self):
        pass
        # ra, dec = self.gal['host']
        # co = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        # result_table = Vizier.query_region(co, radius=0.1 * u.arcmin, catalog='HyperLeda')
        # if (len(result_table) == 0):
        #     self.gal['type'] = 'none'
        #     self.gal['t'] = [-9, 20]
        #     self.gal['incl'] = -1
        #     self.gal['source'] = 'Hyperleda'
        # else:
        #     gal_name = result_table[0]['PGC'][0]
        #     url = f'https://leda.univ-lyon1.fr/ledacat.cgi?PGC{gal_name}&ob=ra'
        #     out = requests.get(url)
        #     soup = bs4.BeautifulSoup(out.content, 'lxml')
        #     table_source = soup.find_all('table')[5]
        #     table = pd.DataFrame(columns=['Parameter', 'Value', 'Unit', 'Description'])
        #     for j in table_source.find_all('tr')[1:]:
        #         row_data = j.find_all('td')
        #         row = [i.text for i in row_data]
        #         table.loc[len(table)] = row

        #     dict_ = dict(zip(table['Parameter'].values, table['Value'].values))
        #     tcode = dict_.get('t', '-9 \pm 20').split()
        #     t = float(tcode[0])
        #     t_err = float(tcode[-1]) if len(tcode) == 3 else 20

        #     self.gal['type'] = dict_.get('type', 'none').strip()
        #     self.gal['source'] = 'hyperleda'
        #     self.gal['t'] = [t, t_err]
        #     self.gal['incl'] = float(dict_.get('incl', '-1').strip())


    def get_galaxy_params(self, source='hl', args=[0, 0, 6]):
        pass
        # def manual(t, gal_type, source):
        #     a, b, t = self.iso.get('a', [args[0]])[0], self.iso.get('b', [args[1]])[0], args[2]
        #     self.gal['type'] = gal_type
        #     self.gal['t'] = [t, 20]
        #     self.gal['incl'] = self.calc_incl(t, a/b)
        #     self.gal['source'] = source

        # if source == 'hl':
        #     self.hyper_leda()
        # elif source == 'calc':
        #     manual(t=args[2], gal_type='none', source='calculated')
        # elif source == 'auto':
        #     self.hyper_leda()
        #     if self.gal['incl'] != -1:
        #         pass
        #     elif (self.gal['incl'] == -1) & (self.gal['t'][0] != -9):
        #         manual(t=self.gal['t'][0], gal_type=self.gal['type'], source='hl_calculated')
        #     else:
        #         manual(t=args[2], gal_type='none', source='calculated')

        # print(self.gal['type'], self.gal['t'], self.gal['incl'], self.gal['source'] ) if self.verbose else None
    
    
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
                return 800
            else:
                aperature = 0.07 # estimate_radius()
                dist = Planck18.luminosity_distance(redshift)
                return min(int(np.rad2deg(aperature/dist.value)*(3600/scale)),  800)
        else:
            return size


    def get_image(self, source, survey, output_, band, scale, soft=True):
        folder_name = 'fits' if survey == 'auto' else survey
        path = f'dr2_{folder_name}_{band}/{self.sn_name}.fits'
        if source == 'query':
            output_size = self.calc_aperature(output_, self.gal['z'], scale)
            fits_ = self.get_cutout(survey, output_size, band, scale)
            fits_data = fits_
        elif source == 'save':
            fits_ = fits.open(path)
            fits_data = fits_
            self.survey = fits_data[1].header['survey']
            output_size = fits_data[1].header['output_size']
        elif source == 'query_save':
            output_size = self.calc_aperature(output_, self.gal['z'], scale)
            fits_ = self.get_cutout(survey, output_size, band, scale)
            fits_data = fits_
            fits_data[1].header['survey'] = self.survey
            fits_data[1].header['output_size'] = output_size
            fits_data.writeto(path, overwrite=True)
            fits_.close()
            return
        else:
            print('invalid source')
            return

        print(self.survey, output_size, scale) if self.verbose else None
        flux, invvar, wcs = fits_data[1].data, fits_data[2].data, WCS(fits_data[1].header)
        exptime = fits_data.header['exptime'] if self.survey == 'ps1' else 1
        mag = self.flux2mag(flux, self.survey, band, scale, soft, exptime=exptime)
        mag_raw = self.flux2mag(flux, self.survey, band, scale, soft=False, exptime=exptime)
        self.cutout = {'flux': flux, 'mag': mag, 'mag_raw': mag_raw, 'invvar': invvar, 'wcs': wcs, 'scale': scale}
        fits_.close()


    def plot(self):
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
        return fig, ax
    


class galaxy_decomp:
    def __init__(self, target_name, verbose, mask, band, source,  size='z', catalog='ztf'):
        self.verbose = verbose
        self.gobj = HostGal(verbose=verbose)
        if catalog == 'ztf':
            self.gobj.init_dr2(target_name)
        else:
            self.gobj.init_query(target_name, catalog)
        self.gobj.get_image(source=source, survey='auto', output_=size, band=band, scale=0.262)

        self.contours = {}
        self.image = self.gobj.cutout['mag']
        self.invvar = self.gobj.cutout['invvar']    
        self.flux = self.gobj.cutout['flux']
        self.mask, self.center = self.generate_masks() if mask else ([], np.array(self.image.shape)//2)

    def plot_fit(self, isophotes, width=0.5, apply_mask=False, zoom=False):
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), dpi=100, ncols=2)
        mag, wcs = self.image, self.gobj.cutout['wcs']
        ax2.imshow(mag, cmap='gray', origin='lower')

        (sn_ra, sn_dec) = self.gobj.gal['sn']
        c = SkyCoord(ra=sn_ra*u.degree, dec=sn_dec*u.degree, frame='icrs')
        sn_px = wcs.world_to_pixel(c)
        ax2.scatter(sn_px[0], sn_px[1], c='cyan', s=40, lw=2, fc='None', ec='cyan')
  
        targets = self.contours.keys() if isophotes == 'all' else isophotes
        for i, iso_i in enumerate(targets):
            params, px_data = self.contours[iso_i]
            a, b, theta, n, xc, yc, *err = params
            px_all, px_fit = px_data

            norm = np.stack(np.where((mag.T > iso_i-width) & (mag.T < iso_i+width))).T
            ax2.scatter(norm.T[0], norm.T[1], s=2, marker='o', zorder=0, color='red', label=f'{iso_i:.1f} $\pm$ {width}')
            if np.any(px_fit):
                ax1.scatter(px_fit.T[0], px_fit.T[1], s=2, marker='o', zorder=1, color='blue', label=f'{iso_i:.1f} fitted')
            if np.any(px_all):
                ax1.scatter(px_all.T[0], px_all.T[1], s=2, marker='o',zorder=0, color='red', label=f'{iso_i:.1f} removed')
           
            if xc != 0:
                self.patch_super_ellipse((a, b, theta, n), (xc, yc), ax1, 'red')
                self.patch_super_ellipse((a, b, theta, n), (xc, yc), ax2, 'lime')
                if zoom:
                    ax1.set_xlim([xc-a-10, xc+a+10])
                    ax2.set_xlim([xc-a-10, xc+a+10])
                    ax1.set_ylim([yc-a-10, yc+a+10])
                    ax2.set_ylim([yc-a-10, yc+a+10])

                else:
                    ax1.set_xlim(*ax2.get_xlim())
                    ax1.set_ylim(*ax2.get_ylim())

        ax1.legend(framealpha=0.8, markerscale=5, loc=2)
        ax2.legend(framealpha=1, markerscale=5, loc=2)
        plt.tight_layout()
        return ax1, ax2
        
    def prep_pixels(self, isophote, blur, window):
        kernal3 = np.array([[ 0,  1,  1],
                            [ -1,  0,  1],
                            [-1,  -1,  0,]])

        kernal_ = 1/(kernal3 + isophote)
        kernal_unit = kernal_ / np.sum(kernal_)
        conv_blur = gaussian_filter(self.image, sigma=blur)
        convolve_1 = convolve2d(conv_blur, kernal_unit, mode='same')
        convolve_2 = convolve2d(convolve_1, kernal_unit[::-1], mode='same')

        convolve_2[:5,:] = 0
        convolve_2[-5:,:] = 0
        convolve_2[:,:5] = 0
        convolve_2[:,-5:] = 0
        
        contour = np.stack(np.where((convolve_2.T > isophote-window) & (convolve_2.T < isophote+window)))
        return contour

    @staticmethod
    def super_ellipse(phi, a_, b_, pa, n, polar=True):
        a, b = max(a_, b_), min(a_, b_)
        def super_transform(phi):
            return np.arctan((a/b)**(n/2) * np.abs(np.tan(phi))**(n/2) * np.sign(np.tan(phi)))

        theta = super_transform(phi-pa)
        x_ = a*np.abs(np.cos(theta))**(2/n)*np.sign(np.cos(theta))  
        y_ = b*np.abs(np.sin(theta))**(2/n)*np.sign(np.sin(theta)) 

        x = x_*np.cos(pa) - y_*np.sin(pa)
        y = x_*np.sin(pa) + y_*np.cos(pa)

        if polar:
            return np.sqrt(x**2 +  y**2)
        else:
            return x, y

    def super_ellipse_fitter(self, data, err, fix_center=True):
        ell = EllipseModel()
        ell.estimate(data)
        if ell.params is None:
            return np.zeros(4), np.zeros(4), (0,0)
        xc, yc, a, b, pa = ell.params
        x, y = data.T
        # if fix_center:
        #     xc, yc = self.center
        r = np.sqrt((x-xc)**2 + (y-yc)**2)
        phi = np.arctan2(y-yc, x-xc)
        out_pars = curve_fit(self.super_ellipse, phi, r, p0=[a, b, pa, 2], maxfev=5000, sigma=err)
        return out_pars[0], np.sqrt(np.diag(out_pars[1])), (xc, yc)

    def patch_super_ellipse(self, pars, center, ax, color, label=None):
        t_r = np.arange(-np.pi/2, np.pi/2, 0.01)+ pars[2]
        xse, yse = self.super_ellipse(t_r, *pars, polar=False) 
        xse_t = np.concatenate([xse, -xse])+ center[0]
        yse_t = np.concatenate([yse, -yse])+ center[1]
        ax.plot(xse_t, yse_t, 'r-', color=color, zorder=10, label=label)
        ax.plot(xse_t[[0, -1]], yse_t[[0, -1]], 'r-', color=color, zorder=10)

    def extract_regions(self, contour):
        def get_pixels(region, connect): return np.stack(np.where(connect == region))

        binary_image = np.zeros_like(self.image, dtype=np.uint8)
        binary_image[contour[0], contour[1]] = 1
        connect_ = ConnectRegion(binary_image, connectivity=2, background=0)
        region_count = np.asarray(np.unique(connect_, return_counts=True)).T[1:].T
        galaxy_region = max(region_count.T, key=lambda x: x[1])[0]
        return get_pixels(galaxy_region, connect_).T         
           
    def contour_fit(self, isophote, window, mask, blur=0):
        all_pixels  = self.prep_pixels(isophote, blur=blur, window=window)
        connect_all = self.extract_regions(all_pixels)

        if len(connect_all) > 25:
            
            # var_mag = -2.5*np.log10(np.sqrt(1/self.invvar)/0.262**2)
            var_frac = np.abs(-2.5/np.log(10)*np.sqrt(1/self.invvar)/self.flux)

            fit_vals = self.image.T[connect_all.T[0], connect_all.T[1]]
            fit_invvar = var_frac.T[connect_all.T[0], connect_all.T[1]]
            cuts = (fit_vals > isophote-window) & (fit_vals < isophote+window)
            connect_all, fit_vals, fit_invvar = connect_all[cuts], fit_vals[cuts], fit_invvar[cuts]

            def linear(x, a, b): return a*x + b
            slope_corr = curve_fit(linear, fit_vals, fit_invvar)
            # (np.abs(fit_vals-isophote))**(1/3) *
            #
            px_uncertainity =  fit_invvar/linear(fit_vals, *slope_corr[0])
            pars, pars_err, center = self.super_ellipse_fitter(connect_all, px_uncertainity)

            self.contours[isophote] = [[*pars, *center, *pars_err, fit_invvar.mean(), fit_vals.mean()], [all_pixels, connect_all]]
            
        else:
            self.contours[isophote] = [[0,0,0,0,0,0,0,0,0,0,0,0], [[], []]]

    def main_run(self):
        step = 0.2
        for iso in np.arange(24.6, 16, -step):
            iso = np.round(iso, 1)
            print(iso) if (iso%1==0.0 and self.verbose) else None
            self.contour_fit(iso, 0.25, mask=True)
            if len(self.contours[iso][1][1]) == 0:
                del self.contours[iso]
                break


class BDdecomp:
    def __init__(self, host_name, gd, remove=None):
        self.host_name = host_name
        self.contours = gd.contours
        self.image = gd.image
        self.invvar = gd.invvar
        self.gal = {'center': gd.center}
        self.mask = gd.mask
        (self.mags, self.iso_data) = self.extract_data(remove)
        self.error_sc = self.scale_error()(self.mags)

    def extract_data(self, remove):
        key_targs = list(self.contours.keys())[slice(remove)]
        mags = np.array(key_targs)
        pars_ = np.array([self.contours[iso_key][0] for iso_key in key_targs if len(self.contours[iso_key][0]) > 0]).T
        
        offsets = np.sqrt((pars_[4]-self.gal['center'][0])**2 + (pars_[5]-self.gal['center'][1])**2)
        cuts = np.where((offsets < 5))

        mags = np.array(key_targs)[cuts]
        pars_ = pars_.T[cuts].T.reshape((12, -1))
        if len(pars_) == 0:
            raise ValueError
        return mags, pars_
    
    @staticmethod
    def super_ellipse(phi, a_, b_, pa, n, polar=True):
        a, b = max(a_, b_), min(a_, b_)
        def super_transform(phi):
            return np.arctan((a/b)**(n/2) * np.abs(np.tan(phi))**(n/2) * np.sign(np.tan(phi)))

        theta = super_transform(phi-pa)
        x_ = a*np.abs(np.cos(theta))**(2/n)*np.sign(np.cos(theta))  
        y_ = b*np.abs(np.sin(theta))**(2/n)*np.sign(np.sin(theta)) 

        x = x_*np.cos(pa) - y_*np.sin(pa)
        y = x_*np.sin(pa) + y_*np.cos(pa)

        if polar:
            return np.sqrt(x**2 +  y**2)
        else:
            return x, y

    def patch_super_ellipse(self, pars, center, ax, color):
        t_r = np.arange(-np.pi/2, np.pi/2, 0.01)+ pars[2]
        xse, yse = self.super_ellipse(t_r, *pars, polar=False) 
        xse_t = np.concatenate([xse, -xse])+ center[0]
        yse_t = np.concatenate([yse, -yse])+ center[1]
        ax.plot(xse_t, yse_t, 'r-', color=color, zorder=10)
        ax.plot(xse_t[[0, -1]], yse_t[[0, -1]], 'r-', color=color, zorder=10)
    
    def plot_iso(self):
        pars_ = self.iso_data
        fig, ax = plt.subplots(figsize=(14, 8), ncols=2, nrows=2, dpi=100)
        ax1, ax2, ax3, ax4 = ax.flatten()

        ecc = 1-pars_[1]/pars_[0]
        ax1.plot(pars_[0]*0.262, ecc, 'r.')
        ax1.set_ylabel('ellipticity')
        ax1.set_xlabel('R [arcsec]')
        ax1.set_ylim([0, 1])

        ax2.plot(pars_[0]*0.262, np.round(pars_[4])+0.02, 'b.', label='center x')
        ax2.plot(pars_[0]*0.262, np.round(pars_[5])-0.02, 'r.', label='center y')
        ax2.set_xlabel('R [arcsec]')
        ax2.legend()

        ax3.plot(pars_[0]*0.262, np.rad2deg(pars_[2]), 'b.')
        ax3.set_ylim([0, 180])
        ax3.set_xlabel('R [arcsec]')
        ax3.set_ylabel('Position angle [deg]')

        ax4.plot(pars_[0]*0.262, pars_[6]*0.262, 'b.', label='fit_err')
        ax4.plot(pars_[0]*0.262, pars_[10], 'r.', label='mag_err')
        xerr = self.iso_data[6]*0.262
        mag_err = self.iso_data[10] 
        yerr = mag_err + xerr
        ax4.plot(pars_[0]*0.262, yerr, 'g.', label='summed err')
        ax4.legend()
        ax4.set_xlabel('R [arcsec]')

        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def get_b(n):
        def func(b, n): return 1 - 2*gammainc(2*n[0], b)
        return fsolve(func, 1.9992*n - 0.3271, args=[n])

    def transform(self, u0, h, n):
        b = np.vectorize(self.get_b)(n)
        return u0 + 2.5*b/np.log(10), b**n*h
    
    def back_transform(self, ue, Re, n):
        b = np.vectorize(self.get_b)(n)
        return ue - 2.5*b/np.log(10), Re/b**n

    def B_D(self, ue, u0, Re, h, n):
        Ie = 10**(-0.4*ue)
        I0 = 10**(-0.4*u0)
        b = np.vectorize(self.get_b)(n)
        return n*gamma(2*n)*np.exp(b)/b**(2*n) * (Re/h)**2 * (Ie/I0)

    def B_T(self, ue, u0, Re, h, n):
        bd = self.B_D(ue, u0, Re, h, n)
        return 1/(1/bd + 1)

    @staticmethod
    def bulge(x, ue, Re, n):
        b = np.vectorize(BDdecomp.get_b)(n)
        return ue + 2.5*b/np.log(10) * ((x/Re)**(1/n) - 1)
    
    @staticmethod
    def disk(x, u0, h):
        return u0 + 2.5/np.log(10)*(x/h)

    @staticmethod
    def add_mag(m1, m2):
        return -2.5*np.log10(10**(-0.4*m1) + 10**(-0.4*m2))

    @staticmethod
    def combine(x, ue, u0, Re, h, n):
        return BDdecomp.add_mag(BDdecomp.bulge(x, ue, Re, n), BDdecomp.disk(x, u0, h))

    @staticmethod
    def combine_2d(pars, x):
        ue, u0, Re, h, n = pars
        # if (u0<18) | (u0>23):
        #     ue *= (1-np.abs(ue-u0)**2)
        return BDdecomp.add_mag(BDdecomp.bulge(x, ue, Re, n), BDdecomp.disk(x, u0, h))
    
    @staticmethod
    def BIC(out, x_data, y_data, model):
        n = len(x_data)
        k = len(out)
        RSS = np.sum((model(x_data, *out) - y_data)**2)
        return n * np.log(RSS/n) + k * np.log(n), RSS, k, n

    def target_angle(self, c_r, theta):
        target_ang = np.zeros(len(c_r))
        for i, row_i in enumerate(c_r):
            ai, bi, thetai, ni, xci, yci, *errs = row_i # fix
            ep = self.super_ellipse(theta, ai, bi, thetai, ni, polar=False)
            target_ang[i] = np.sqrt((ep[0])**2 + (ep[1])**2)
        return target_ang * 0.262

    def plot_gal_iso(self, theta=None, lims=[], ax=[]):
        if ax:
            fig, ax = '', ax
        else:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(self.image, cmap='gray', origin='lower')
        if len(lims) == 0:
            lims = [0, len(self.image)]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
  
        for row_i in self.iso_data.T[::-2]:
            ai, bi, thetai, ni, xci, yci, *errs = row_i
            if theta is not None:
                xp, yp = self.super_ellipse(np.deg2rad(theta), ai, bi, thetai, ni, polar=False)
                ax.scatter(xp+xci, yp+yci, color='blue', s=20, zorder=15)

            self.patch_super_ellipse((ai, bi, thetai, ni), (xci, yci), ax, 'lime')
        return fig, ax
    
    @staticmethod
    def ellipse_para(theta, xc, yc, a, b, pa):
        theta = np.arctan(a/b*np.tan(theta))
        x = a*np.cos(theta)*np.cos(pa) - b*np.sin(theta)*np.sin(pa) + xc
        y = a*np.cos(theta)*np.sin(pa) + b*np.sin(theta)*np.cos(pa) + yc
        return x, y
    
    def scale_error(self):
        mag_trim = self.image.flatten()[self.image.flatten()<25]
        inv_trim = self.invvar.flatten()[self.image.flatten()<25]
        mag_bins = np.round(np.arange(np.round(mag_trim.min(), 1), 25, 0.1), 1)

        bins = np.digitize(mag_trim, mag_bins)
        inv_bins = [np.mean(inv_trim[np.where(bins==i)[0]]) for i in range(len(mag_bins))]
        scale_inv =  inv_bins/inv_trim.max()
        fit_interp = interp1d(mag_bins, np.sqrt(scale_inv))
        return fit_interp

    def fit_profile(self, theta, plot=False):

        x_data = self.target_angle(self.iso_data.T, np.deg2rad(theta))
        xerr = self.iso_data[6]*0.262
        mag_err = self.iso_data[10] 
        # yerr= np.max(np.stack([mag_err, 3*xerr]), axis=0)
        yerr = mag_err + xerr
        # yerr=mag_err

        mags_ = self.iso_data[11] 
        max_r = len(self.image)/2 * 0.262


        try:
            fit_2model, fit_2errs = curve_fit(self.combine, x_data, mags_, sigma=yerr, p0=[19, 20, 6, 10, 3], 
                                             maxfev=5000, bounds=[[17,18,0.3,0.3,0.1], [27,27,max_r,max_r,15]])
            fit_2errs = np.sqrt(np.diag(fit_2errs))

            # func_ = Model(self.combine_2d)
            # mydata = Data(x_data, mags_, wd=xerr, we=yerr)
            # beta0 = [20.8612786 , 23,  1.23825444,  4.1262658 ,  0.85353317]
            # myodr = ODR(mydata, func_, beta0=beta0)
            # myoutput = myodr.run()
            # fit_2model, fit_2errs = myoutput.beta, myoutput.sd_beta

        except RuntimeError:
            fit_2model, fit_2errs = np.zeros(5), np.zeros(5)

        try:
            fit_1model, fit_1errs = curve_fit(self.bulge, x_data, mags_, sigma=yerr, p0=[19, 10, 4], maxfev =5000)
            fit_1errs = np.sqrt(np.diag(fit_1errs))
        except RuntimeError:
            fit_1model, fit_1errs = np.zeros(3), np.zeros(3)

        if plot:
            fig, (ax1, ax2) = plt.subplots(figsize=(12, 10), dpi=100, nrows=2)
            x_ax = np.linspace(np.min(x_data), np.max(x_data), 100)
            ax1.errorbar(x_data, self.mags, yerr=yerr, fmt='k.', zorder=0)
            # ax1.plot(x_data, self.mags, 'k.')
            ax1.plot(x_ax, self.combine(x_ax, *fit_2model), 'r-', label='combined')
            ue, u0, Re, h, n = fit_2model
            ax1.plot(x_ax, self.bulge(x_ax, ue, Re, n), 'g--', label='bulge')
            ax1.plot(x_ax, self.disk(x_ax, u0, h), 'b--', label='disk')

            ax1.set_ylim([self.mags.min()-0.5, 25.5])
            ax1.invert_yaxis()
            ax1.legend(fontsize=12)
            ax1.set_xlabel('R [arcsec]', fontsize=12)
            ax1.set_ylabel('$\mu \:[mag\:arcsec^{-1}]$', fontsize=12)

            ax2.errorbar(x_data, self.mags, yerr=yerr, fmt='k.', zorder=0)
            # ax2.plot(x_data, self.mags, 'k.')
            ax2.plot(x_ax, self.bulge(x_ax, *fit_1model), 'r-', label='bulge')

            ax2.invert_yaxis()
            ax2.legend(fontsize=12)
            ax2.set_xlabel('R [arcsec]', fontsize=12)
            ax2.set_ylabel('$\mu \:[mag\:arcsec^{-1}]$', fontsize=12)
            return fig, (ax1, ax2), (fit_2model, fit_1model, fit_2errs, fit_1errs)
        else:
            return fit_2model, fit_1model, fit_2errs, fit_1errs
            
    def bulge_disk_2d(self):
        ang_r = np.arange(0, 180, 5)
        bd_res = np.zeros((len(ang_r),14))
        for i, theta_i in enumerate(ang_r):
            out2_i, out1_i, out2_ierr, out1_i_err = self.fit_profile(theta_i, plot=False)
            ue_b2, u0_d2, Re_b2, h_d2, n2 = out2_i
            ue_d2, Re_d2 = self.transform(u0_d2, h_d2, n=1)
            ue_b1, Re_b1, n1 = out1_i
            BT = self.B_T(*out2_i)
            bd_res[i] = [ue_d2, Re_d2, ue_b2, Re_b2, n2, ue_b1, Re_b1, n1, BT, *out2_ierr]
        
        self.decomp = bd_res


class galaxy_decomp_old:
    def __init__(self, target_name, verbose, mask, band, source,  size='z', catalog='ztf'):
        self.verbose = verbose
        self.gobj = HostGal(verbose=verbose)
        if catalog == 'ztf':
            self.gobj.init_dr2(target_name)
        else:
            self.gobj.init_query(target_name, catalog)
        self.gobj.get_image(source=source, survey='auto', output_=size, band=band, scale=0.262)

        self.contours = {}
        self.image = self.gobj.cutout['mag']
        self.invvar = self.gobj.cutout['invvar']
        self.mask, self.center = self.generate_masks() if mask else ([], np.array(self.image.shape)//2)
        

    def plot_fit(self, isophotes, width=0.5, apply_mask=False):
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), dpi=100, ncols=2)
        mag, wcs = self.image, self.gobj.cutout['wcs']
        ax2.imshow(mag, cmap='gray', origin='lower')

        (sn_ra, sn_dec) = self.gobj.gal['sn']
        c = SkyCoord(ra=sn_ra*u.degree, dec=sn_dec*u.degree, frame='icrs')
        sn_px = wcs.world_to_pixel(c)
        ax2.scatter(sn_px[0], sn_px[1], c='cyan', s=40, lw=2, fc='None', ec='cyan')
        if np.any(self.mask) & apply_mask:
            for (x, y, a, b, theta) in self.mask:
                m_patch = Ellipse((x, y), 2*a, 2*b, np.rad2deg(theta), edgecolor='black', facecolor='black', zorder=10, alpha=0.5)
                ax1.add_patch(m_patch)

        targets = self.contours.keys() if isophotes == 'all' else isophotes
        for i, iso_i in enumerate(targets):
            params, px_data = self.contours[iso_i]
            xc, yc, a, b, theta, xc_err, yc_err, a_err, b_err, theta_err = params
            # xc, yc, a, b, theta, chi2, wi = params
            px_all, px_fit = px_data

            norm = np.stack(np.where((mag.T > iso_i-width) & (mag.T < iso_i+width))).T
            ax2.scatter(norm.T[0], norm.T[1], s=1, marker='o', zorder=0, label=f'{iso_i:.1f} $\pm$ {width}')
            if np.any(px_fit):
                ax1.scatter(px_fit.T[0], px_fit.T[1], s=1, marker='o', zorder=1, label=f'{iso_i:.1f} fitted')
            if np.any(px_all):
                ax1.scatter(px_all.T[0], px_all.T[1], s=1, marker='o',zorder=0, label=f'{iso_i:.1f} removed')
           
            if xc != 0:
                ell_patch1 = Ellipse((xc, yc), 2*a, 2*b, np.rad2deg(theta), edgecolor='black', facecolor='none')
                ax1.add_patch(ell_patch1)
                ell_patch2 = Ellipse((xc, yc), 2*a, 2*b, np.rad2deg(theta), edgecolor='lime', facecolor='none')
                ax2.add_patch(ell_patch2)

                ax1.set_xlim(*ax2.get_xlim())
                ax1.set_ylim(*ax2.get_ylim())
        ax1.legend(framealpha=0.8, markerscale=5, loc=2)
        ax2.legend(framealpha=1, markerscale=5, loc=2)
        plt.tight_layout()
        return ax1, ax2


    def prep_pixels(self, isophote, blur, window, split=False):
        kernal5 = np.array([[ 0,  0,  1,  1,  1],
                            [ 0,  0,  0,  1,  1],
                            [-1,  0,  0,  0,  1],
                            [-1, -1,  0,  0,  0],
                            [-1, -1, -1,  0,  0]])
        kernal3 = np.array([[ 0,  1,  1],
                            [ -1,  0,  1],
                            [-1,  -1,  0,]])

        kernal_ = 1/(kernal3 + isophote)
        kernal_unit = kernal_ / np.sum(kernal_)
        blur = 1 if isophote < 24 else 3
        conv_blur = gaussian_filter(self.image, sigma=blur)
        convolve_1 = convolve2d(conv_blur, kernal_unit, mode='same')
        convolve_2 = convolve2d(convolve_1, kernal_unit[::-1], mode='same')

        convolve_2[:5,:] = 0
        convolve_2[-5:,:] = 0
        convolve_2[:,:5] = 0
        convolve_2[:,-5:] = 0
        
        def get_target(iso):
            Y, X = np.ogrid[:len(self.image), :len(self.image)]
            dist_from_center = np.sqrt((X - self.center[0])**2 + (Y - self.center[1])**2)
            mask = dist_from_center <= 100
            cmag = convolve_2 * mask
            min_val = cmag[~np.isinf(1/cmag)].min()
            slope = len(cmag)/(25 - min_val)/2
            return slope * (iso - min_val)

        if split:
            for wi in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                contour = np.stack(np.where((convolve_2.T > isophote-wi) & (convolve_2.T < isophote+wi)))
                if len(contour.T) > 2*np.pi*get_target(isophote)*10:
                    break
            window = wi

        contour = np.stack(np.where((convolve_2.T > isophote-window) & (convolve_2.T < isophote+window)))
        contour_l = np.stack(np.where((convolve_2.T > isophote-window) & (convolve_2.T < isophote)))
        contour_u = np.stack(np.where((convolve_2.T > isophote) & (convolve_2.T < isophote+window)))
        return contour.T if not split else [contour.T, contour_l.T, contour_u.T]


    def ellipse_fit(self, px_d):
        if len(px_d) == 0:
            return (0,0,0,0,0), 0
        ell = EllipseModel()
        ell.estimate(px_d)
        if ell.params is None:
            return (0,0,0,0,0), 0
        else:
            res = ell.residuals(px_d)
            chi2 = np.sum(res**2)/(len(res)-len(ell.params))
            return ell.params, chi2
        
    def generate_masks(self):
        def get_pixels(region, connect): return np.stack(np.where(connect == region))
        def linear_fit(x, a, b): return a*x + b
        def inside_ellipse(x, y, params):
            xc, yc, a, b, theta = params
            term1 = np.cos(theta)*(x - xc) + np.sin(theta)*(y - yc)
            term2 = np.sin(theta)*(x - xc) - np.cos(theta)*(y - yc)
            return term1**2/a**2 + term2**2/b**2 <= 1

        contour_data, astro_objects, obj_centers = {}, {}, []

        for iso_i in np.arange(24.5, 17, -0.5):
            contour_i = self.prep_pixels(iso_i, blur=1, window=1)
            if len(contour_i) == 0:
                continue

            binary_image = np.zeros_like(self.image, dtype=np.uint8)
            binary_image[contour_i.T[0], contour_i.T[1]] = 1
            connect_ = ConnectRegion(binary_image, connectivity=2, background=0)
            region_count = np.asarray(np.unique(connect_, return_counts=True))
            cond1 = (region_count[1] >= region_count[1].max()*0.01) | (region_count[1] > 20)
            region_count = region_count.T[cond1 & (region_count[0] != 0)].T

            iso_objects = np.zeros((len(region_count[0]), 6))
            for i, rc in enumerate(region_count[0]):
                px_ = get_pixels(rc, connect_)
                pars, chi2 = self.ellipse_fit(px_.T[::max(len(px_.T)//3000, 1)])
                iso_objects[i] = np.round([*pars, chi2], 2)
            if len(iso_objects.T[0]) > 0:
                contour_data[iso_i] = iso_objects.T
        
        for iso_level in list(contour_data.keys())[::-1]:
            mask_index = np.arange(len(obj_centers))
            for obj in contour_data[iso_level].T:
                identified = False
                xc_i, yc_i, a_i, b_i, theta_i, chi2_ = obj
                for i, (cx, cy) in enumerate(obj_centers):
                    offset = np.sqrt((xc_i-cx)**2 + (yc_i-cy)**2)
                    if offset < 10:
                        astro_objects[i]['a'].append([iso_level, a_i])
                        astro_objects[i]['b'].append([iso_level, b_i])
                        astro_objects[i]['theta'].append([iso_level, theta_i])
                        identified = True
                        mask_index = np.delete(mask_index, np.where(mask_index == i)[0])
                        break
                if not identified:
                    merged_center = False
                    for obj_key in astro_objects.keys():
                        obj = astro_objects[obj_key]
                        cx, cy = obj_centers[obj_key]
                        a, b, theta = obj['a'][-1][1], obj['b'][-1][1], obj['theta'][-1][1]
                        if inside_ellipse(xc_i, yc_i, [cx, cy, a, b, theta]):
                            merged_center = True
                            break

                    if not merged_center: 
                        obj_centers.append([xc_i, yc_i])
                        astro_objects[len(astro_objects)] = {'a': [[iso_level, a_i]], 'b': [[iso_level, b_i]],
                                                'theta': [[iso_level, theta_i]]}
        
        if len(obj_centers) == 0:
            return [], np.array(self.image.shape)//2
        else:
            dists = np.linalg.norm(np.array(obj_centers) - np.array(self.image.shape)//2, axis=1)
            sizes = [astro_objects[i]['a'][-1][1] for i in astro_objects.keys()]
            gi_close, gi_size = np.argmin(dists), np.argmax(sizes)
            if gi_close != gi_size:
                if sizes[gi_size] > sizes[gi_close] * 5:
                    galaxy_i = gi_size
                else:
                    galaxy_i = gi_close
            else:
                galaxy_i = gi_close
            mask_index = np.delete(mask_index, np.where(mask_index == galaxy_i)[0])

            masks = np.zeros((len(mask_index), 5))
            for i, mi in enumerate(mask_index):
                mx, my = obj_centers[mi]
                a_m = np.array(astro_objects[mi]['a'])
                b_m = np.array(astro_objects[mi]['b'])
                theta_m = astro_objects[mi]['theta'][-1][1]
                if not inside_ellipse(*obj_centers[galaxy_i], [mx, my, a_m[-1][1], b_m[-1][1], theta_m]):
                    if len(a_m)>2:
                        out_ma = curve_fit(linear_fit, a_m.T[0], a_m.T[1])
                        out_mb = curve_fit(linear_fit, b_m.T[0], b_m.T[1])
                        ai, bi = linear_fit(25, *out_ma[0])*1.5, linear_fit(25, *out_mb[0])*1.5
                    else:
                        ai, bi = a_m[-1][1]*(3+(24-a_m[-1][0])/2), b_m[-1][1]*(3+(24-a_m[-1][0])/2)
                    masks[i] = [mx, my, ai, bi, theta_m]
            masks = masks[~np.all(masks == 0, axis=1)]
            offset = np.linalg.norm(np.array(self.image.shape)//2 - obj_centers[galaxy_i])
            print(f'masks: {len(masks)}, offset: {offset:.0f}px') if self.verbose else None
            return masks, obj_centers[galaxy_i]

    def extract_region(self, isophote_data, apply_mask=False):
        def get_pixels(region, connect): return np.stack(np.where(connect == region))

        binary_image = np.zeros_like(self.image, dtype=np.uint8)
        binary_image[isophote_data.T[0], isophote_data.T[1]] = 1
        connect_ = ConnectRegion(binary_image, connectivity=2, background=0)
        region_count = np.asarray(np.unique(connect_, return_counts=True))
        region_count = region_count.T[(region_count[1] > 10) & (region_count[0] != 0)].T
        if len(region_count.T) == 0:
            return []
        else:
            region_offset = np.zeros_like(region_count[0])
            for i, reg_i in enumerate(region_count[0]):
                px_i = get_pixels(reg_i, connect_)
                px_center = np.mean(px_i, axis=1)
                region_offset[i] = np.linalg.norm(px_center-self.center)
            galaxy_region = region_count[0][np.argmin(region_offset)]
            
            if np.any(self.mask) & apply_mask:
                for (x, y, a, b, ang) in self.mask:
                    ell_px = draw.ellipse(x, y, a, b, rotation=ang)
                    mask = np.ones_like(self.image)
                    px_cutoff = (ell_px[0] < self.image.shape[0]) & (ell_px[1] < self.image.shape[0])
                    mask[ell_px[0][px_cutoff], ell_px[1][px_cutoff]] = 0
                    connect_ = connect_ * mask

            return get_pixels(galaxy_region, connect_).T
    
    def trim_pixels(self, data, data_low, data_upp):

        def coord_in(x, y):
            return np.where((data.T[0] == x) & (data.T[1] == y))

        def polar_transform(xi, yi):
            vec0 = np.array([-1, 0])
            veci = np.array([xi, yi]) - np.mean(data, axis=0)
            ni = np.linalg.norm(veci)
            ang = np.arccos(np.dot(veci, vec0)/ni)
            sign = -np.sign(np.arcsin(np.cross(veci,vec0)/ni))
            return sign*ang, ni

        def unravel(data):
            res_thetas = np.vectorize(polar_transform)(data.T[0], data.T[1])
            res_thetas = res_thetas + abs(np.min(res_thetas))
            return res_thetas
        
        def boundary(x):
            theta_bin_low = yl[np.where(xlr == x)]
            # theta_bin_upp = yu[np.where(xur == x)]
            return theta_bin_low.max()
        
        px_unwrapped = unravel(data)
        ind_l = [int(coord_in(x, y)[0]) for x, y in data_low]
        ind_u = [int(coord_in(x, y)[0]) for x, y in data_upp]
        xu, yu = px_unwrapped[0][ind_u], px_unwrapped[1][ind_u]
        xl, yl = px_unwrapped[0][ind_l], px_unwrapped[1][ind_l]

        xur, xlr = np.round(px_unwrapped[0][ind_u], 1), np.round(px_unwrapped[0][ind_l], 1)
        r_ang = np.unique(np.intersect1d(xur, xlr))
        if (len(r_ang) == 0) or (len(px_unwrapped[0]) == 0):
            return [], 0

        vals = np.vectorize(boundary)(r_ang)
        central_line = interp1d(r_ang, vals, bounds_error=False)(px_unwrapped[0])
        res = px_unwrapped[1] - central_line
        r = central_line[~np.isnan(central_line)]
        if len(r) == 0:
            return [], 0
        
        select = np.arange(int(2*np.pi*r.mean()) * 4)
        if len(res) < len(select):
             return [], 0

        px_window = np.argpartition(np.abs(res), select)[select]
        px_mirror = res[px_window] < np.abs(np.min(res[~np.isnan(res)]))
        fit_window = res[px_window][px_mirror].max()

        return data[px_window][px_mirror], fit_window

    def small_isophote(self, isophote, fit_pixels, fit_window):
        if len(fit_pixels) > 200:
            return fit_pixels, fit_window
        else:
            Y, X = np.ogrid[:len(self.image), :len(self.image)]
            dist_from_center = np.sqrt((X - self.center[0])**2 + (Y - self.center[1])**2)
            mask = dist_from_center <= 50
            mag = self.image * mask
            mag_px = mag[~np.isinf(1/mag)]
            fit_window = abs(isophote - mag_px.min())/3
            fit_pixels = np.stack(np.where((mag.T > isophote-fit_window) &
                                            (mag.T < isophote+fit_window)))
            return fit_pixels.T, 1
        
    def fit_algorithm(self, data, data_low, data_upp, isophote):
        def coord_in(x, y):
            return np.where((data.T[0] == x) & (data.T[1] == y))

        def polar_transform(xi, yi):
            vec0 = np.array([-1, 0])
            veci = np.array([xi, yi]) - np.mean(data, axis=0)
            ni = np.linalg.norm(veci)
            ang = np.arccos(np.dot(veci, vec0)/ni)
            sign = -np.sign(np.arcsin(np.cross(veci, vec0)/ni))
            return sign*ang, ni

        def unravel(data):
            res_thetas = np.vectorize(polar_transform)(data.T[0], data.T[1])
            res_thetas = res_thetas + abs(np.min(res_thetas))
            return res_thetas
        
        def boundary(x):
            theta_bin_low = yl[np.where(xlr == x)]
            return theta_bin_low.max()
        
        px_unwrapped = unravel(data)
        ind_l = [int(coord_in(x, y)[0]) for x, y in data_low]
        ind_u = [int(coord_in(x, y)[0]) for x, y in data_upp]
        xu, yu = px_unwrapped[0][ind_u], px_unwrapped[1][ind_u]
        xl, yl = px_unwrapped[0][ind_l], px_unwrapped[1][ind_l]

        xur, xlr = np.round(xu, 1), np.round(xl, 1)
        r_ang = np.unique(np.intersect1d(xur, xlr))
        if (len(r_ang) == 0) or (len(px_unwrapped[0]) == 0):
            return [], [], []

        vals = np.vectorize(boundary)(r_ang)
        central_line = interp1d(r_ang, vals, bounds_error=False)(px_unwrapped[0])
        res = px_unwrapped[1] - central_line
        r = central_line[~np.isnan(central_line)]
        if len(r) == 0:
            return [], [], []

        window_range = [2, 2.5, 3, 4, 4.5, 5, 3.5]
        fit_pars = np.zeros((len(window_range), 5))
        fit_errs = np.zeros(len(window_range))
        for i, wi in enumerate(window_range):
            select = np.arange(int(2*np.pi*r.mean() * wi))
            if len(res) < len(select):
                fit_pixels = []
            else:
                px_window = np.argpartition(np.abs(res), select)[select]
                px_mirror = res[px_window] < np.abs(np.min(res[~np.isnan(res)]))
                fit_pixels = data[px_window][px_mirror]
                fit_pars[i], fit_errs[i] = self.ellipse_fit(fit_pixels)
                
        if (len(fit_pixels) > 250):
            pars = np.average(fit_pars, axis=0, weights=1/fit_errs)
            pars_err = np.std(fit_pars, axis=0)/np.sum(fit_errs!=0)
            return pars, pars_err, fit_pixels
        else:
            Y, X = np.ogrid[:len(self.image), :len(self.image)]
            dist_from_center = np.sqrt((X - self.center[0])**2 + (Y - self.center[1])**2)
            mask = dist_from_center <= 50
            mag = self.image * mask
            mag_px = mag[~np.isinf(1/mag)]
            fit_pars = np.zeros((len(window_range), 5))
            fit_errs = np.zeros(len(window_range))
            for i, wi in enumerate(window_range):
                fit_window = abs(isophote - mag_px.min())/wi
                fit_pixels = np.stack(np.where((mag.T > isophote-fit_window) &
                                                (mag.T < isophote+fit_window))).T
                fit_pars[i], fit_errs[i] = self.ellipse_fit(fit_pixels)

            if (np.sum(fit_pars) != 0) and (np.sum(fit_errs) != 0):
                pars = np.average(fit_pars, axis=0, weights=1/fit_errs)
                pars_err = np.std(fit_pars, axis=0)/np.sum(fit_errs!=0)
                return pars, pars_err, fit_pixels
            else:
                return [], [], []
            
           
    def contour_fit(self, isophote, mask=False):
        all_pixels, low_pixels, upp_pixels = self.prep_pixels(isophote, blur=1, window=0.5, split=True)
        connect_all = self.extract_region(all_pixels, apply_mask=mask)
        connect_low = self.extract_region(low_pixels, apply_mask=mask)
        connect_upp = self.extract_region(upp_pixels, apply_mask=mask)
        if (len(connect_all) >= len(connect_low) + len(connect_upp)) and  (len(connect_all) != 0):
            # fit_pixels, fit_window = self.trim_pixels(connect_all, connect_low, connect_upp)
            # fit_pixels, fit_window = self.small_isophote(isophote, fit_pixels, fit_window)
            # pars, chi2 = self.ellipse_fit(fit_pixels)
            # self.contours[isophote] = [[*pars, chi2, fit_window], [all_pixels, fit_pixels]]

            pars, pars_err, fit_pixels = self.fit_algorithm(connect_all, connect_low, connect_upp, isophote)
            self.contours[isophote] = [[*pars, *pars_err], [all_pixels, fit_pixels]]
            
        else:
            # self.contours[isophote] = [[0,0,0,0,0,0,0], [[], []]]
            self.contours[isophote] = [[0,0,0,0,0,0,0,0,0,0], [[], []]]


    def main_run(self):
        Y, X = np.ogrid[:len(self.image), :len(self.image)]
        dist_from_center = np.sqrt((X - self.center[0])**2 + (Y - self.center[1])**2)
        mask = dist_from_center <= 50
        mag = self.image * mask
        mag = mag[~np.isinf(1/mag)]
        for iso in np.arange(24.6, mag.min()-1, -0.2):
            iso = np.round(iso, 1)
            print(iso) if (iso%1==0.0 and self.verbose) else None
            self.contour_fit(iso, mask=True)
            if len(self.contours[iso][1][0]) == 0:
                del self.contours[iso]


class BDdecomp_old:
    def __init__(self, host_name, contours, image, center, mask, remove=None):
        self.host_name = host_name
        self.contours = contours
        self.image = image
        self.gal = {'center': center}
        self.mask = mask
        (self.mags, self.iso_data) = self.extract_data(remove)

    def extract_data(self, remove):
        key_targs = list(self.contours.keys())[slice(remove)]
        pars_ = np.array([self.contours[iso_key][0] for iso_key in key_targs if len(self.contours[iso_key][0]) > 0]).T
        if len(pars_) == 0:
            raise ValueError
        offsets = np.sqrt((pars_[0]-self.gal['center'][0])**2 + (pars_[1]-self.gal['center'][1])**2)
        cuts = np.where((pars_[-1] !=0) & (~np.isnan(pars_[-1]) & (offsets < 10) & (pars_[5] > 0)))

        mags = np.array(key_targs)[cuts]
        pars_ = pars_.T[cuts].T.reshape((10, -1))
        self.gal['disk_ell'] = np.average((1-pars_[3]/pars_[2])[:len(mags)//3], weights = 1/pars_[5][:len(mags)//3])
        self.gal['pa'] = np.average(np.rad2deg(pars_[4])[:len(mags)//3], weights = 1/pars_[5][:len(mags)//3])
        return mags, pars_
    
    def plot_iso(self):
        pars_ = self.iso_data
        fig, ax = plt.subplots(figsize=(14, 8), ncols=2, nrows=2, dpi=100)
        ax1, ax2, ax3, ax4 = ax.flatten()

        ecc = 1-pars_[3]/pars_[2]
        ax1.plot(pars_[2]*0.262, ecc, 'r.')
        ax1.axhline(self.gal['disk_ell'], color='k', label='disk ellipticity')
        ax1.set_ylabel('ellipticity')
        ax1.set_xlabel('R [arcsec]')
        ax1.set_ylim([0, 1])
        ax1.legend()

        ax2.plot(pars_[2]*0.262, np.round(pars_[0])+0.02, 'b.', label='center x')
        ax2.plot(pars_[2]*0.262, np.round(pars_[1])-0.02, 'r.', label='center y')
        ax2.set_xlabel('R [arcsec]')
        ax2.legend()

        ax3.plot(pars_[2]*0.262, np.rad2deg(pars_[4]), 'b.')
        ax3.set_ylim([0, 180])
        ax3.axhline(self.gal['pa'], color='k', label='disk position angle')
        ax3.legend()
        ax3.set_xlabel('R [arcsec]')
        ax3.set_ylabel('Position angle [deg]')

        ax4.plot(pars_[2]*0.262, pars_[7], 'b.', label='$\chi^2$')
        ax4.plot(pars_[2]*0.262, pars_[8], 'r.', label='fit window [px]')
        ax4.plot(pars_[2]*0.262, (pars_[7]+pars_[8])/2, 'g.', label='err')
        ax4.legend()
        ax4.set_xlabel('R [arcsec]')

        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def get_b(n):
        def func(b, n): return 1 - 2*gammainc(2*n[0], b)
        return fsolve(func, 1.9992*n - 0.3271, args=[n])

    def transform(self, u0, h, n):
        b = np.vectorize(self.get_b)(n)
        return u0 + 2.5*b/np.log(10), b**n*h
    
    def back_transform(self, ue, Re, n):
        b = np.vectorize(self.get_b)(n)
        return ue - 2.5*b/np.log(10), Re/b**n

    def B_D(self, ue, u0, Re, h, n):
        Ie = 10**(-0.4*ue)
        I0 = 10**(-0.4*u0)
        b = np.vectorize(self.get_b)(n)
        return n*gamma(2*n)*np.exp(b)/b**(2*n) * (Re/h)**2 * (Ie/I0)

    def B_T(self, ue, u0, Re, h, n):
        bd = self.B_D(ue, u0, Re, h, n)
        return 1/(1/bd + 1)

    @staticmethod
    def bulge(x, ue, Re, n):
        b = np.vectorize(BDdecomp.get_b)(n)
        return ue + 2.5*b/np.log(10) * ((x/Re)**(1/n) - 1)
    
    @staticmethod
    def disk(x, u0, h):
        return u0 + 2.5/np.log(10)*(x/h)

    @staticmethod
    def add_mag(m1, m2):
        return -2.5*np.log10(10**(-0.4*m1) + 10**(-0.4*m2))

    @staticmethod
    def combine(x, ue, u0, Re, h, n):
        return BDdecomp.add_mag(BDdecomp.bulge(x, ue, Re, n), BDdecomp.disk(x, u0, h))
    
    @staticmethod
    def BIC(out, x_data, y_data, model):
        n = len(x_data)
        k = len(out)
        RSS = np.sum((model(x_data, *out) - y_data)**2)
        return n * np.log(RSS/n) + k * np.log(n), RSS, k, n

    @staticmethod
    def ellipse_para(theta, xc, yc, a, b, pa):
        theta = np.arctan(a/b*np.tan(theta))
        x = a*np.cos(theta)*np.cos(pa) - b*np.sin(theta)*np.sin(pa) + xc
        y = a*np.cos(theta)*np.sin(pa) + b*np.sin(theta)*np.cos(pa) + yc
        return x, y

    def target_angle(self, c_r, theta_ref):
        target_ang = np.zeros(len(c_r))
        for i, row_i in enumerate(c_r):
            xci, yci, ai, bi, thetai, *errs = row_i
            ep = self.ellipse_para((theta_ref - thetai),  xci, yci, ai, bi, thetai)
            target_ang[i] = np.sqrt((ep[0]-xci)**2 + (ep[1]-yci)**2)
        return target_ang * 0.262
    
    def plot_gal_iso(self, theta='sma', lims=[], ax=[]):
        theta_ref = np.deg2rad(self.gal['pa']) if theta=='sma' else np.deg2rad(theta)
        if ax:
            fig, ax = '', ax
        else:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(self.image, cmap='gray', origin='lower')
        if len(lims) == 0:
            lims = [0, len(self.image)]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        for (x, y, a, b, theta) in self.mask:
                m_patch = Ellipse((x, y), 2*a, 2*b, np.rad2deg(theta), facecolor='red', zorder=10, alpha=0.5)
                ax.add_patch(m_patch)
        for row_i in self.iso_data.T[::-2]:
            xci, yci, ai, bi, thetai, *errs = row_i
            ep1 = self.ellipse_para(theta_ref - thetai,  xci, yci, ai, bi, thetai)
            ax.scatter(ep1[0], ep1[1], color='blue', s=20, zorder=10)
            ell_patchi = Ellipse((xci, yci), 2*ai, 2*bi, np.rad2deg(thetai), edgecolor='lime', facecolor='none', lw=1)
            ax.add_patch(ell_patchi)
        return fig, ax

    def fit_profile(self, theta='sma', plot=False):
        theta_ref = np.deg2rad(self.gal['pa']) if theta=='sma' else np.deg2rad(theta)
        x_data = self.target_angle(self.iso_data.T, theta_ref)
        err = self.iso_data[8]
        max_r = len(self.image)/2 * 0.262
        try:
            fit_2model, fit_errs = curve_fit(self.combine, x_data, self.mags, p0=[19, 20, 6, 10, 3], 
                                             maxfev=5000, bounds=[[17,18,0.3,0.3,0.1], [27,27,max_r,max_r,15]])
        except RuntimeError:
            fit_2model = np.zeros(5)

        try:
            fit_1model, fit_errs = curve_fit(self.bulge, x_data, self.mags, sigma=err, p0=[19, 10, 4], maxfev =5000)[0]
        except RuntimeError:
            fit_1model = np.zeros(3)
        if plot:
            fig, (ax1, ax2) = plt.subplots(figsize=(12, 10), dpi=100, nrows=2)
            x_ax = np.linspace(np.min(x_data), np.max(x_data), 100)

            ax1.errorbar(x_data, self.mags, yerr=err, fmt='k.', zorder=0)
            # ax1.plot(x_data, self.mags, 'k.')
            ax1.plot(x_ax, self.combine(x_ax, *fit_2model), 'r-', label='combined')
            ue, u0, Re, h, n = fit_2model
            ax1.plot(x_ax, self.bulge(x_ax, ue, Re, n), 'g--', label='bulge')
            ax1.plot(x_ax, self.disk(x_ax, u0, h), 'b--', label='disk')

            ax1.set_ylim([self.mags.min()-0.5, 25.5])
            ax1.invert_yaxis()
            ax1.legend(fontsize=12)
            ax1.set_xlabel('R [arcsec]', fontsize=12)
            ax1.set_ylabel('$\mu \:[mag\:arcsec^{-1}]$', fontsize=12)

            ax2.errorbar(x_data, self.mags, yerr=err, fmt='k.', zorder=0)
            # ax2.plot(x_data, self.mags, 'k.')
            ax2.plot(x_ax, self.bulge(x_ax, *fit_1model), 'r-', label='bulge')

            ax2.invert_yaxis()
            ax2.legend(fontsize=12)
            ax2.set_xlabel('R [arcsec]', fontsize=12)
            ax2.set_ylabel('$\mu \:[mag\:arcsec^{-1}]$', fontsize=12)
            return fig, (ax1, ax2)
        else:
            return fit_2model, fit_1model, x_data
            
    def bulge_disk_2d(self):
        ang_r = np.arange(0, 180, 5)
        ang_r = np.append(ang_r, self.gal['pa'])
        bd_res = np.zeros((len(ang_r),13))
        for i, theta_i in enumerate(ang_r):
            out2_i, out1_i, x_data_i = self.fit_profile(theta_i, plot=False)
            ue_b2, u0_d2, Re_b2, h_d2, n2 = out2_i
            ue_d2, Re_d2 = self.transform(u0_d2, h_d2, n=1)
            ue_b1, Re_b1, n1 = out1_i
            BIC_2 = self.BIC(out2_i, x_data_i, self.mags, self.combine)[0]
            BIC_1 = self.BIC(out1_i, x_data_i, self.mags, self.bulge)[0]
            BT = self.B_T(*out2_i)
            
            bd_res[i] = [ue_d2, Re_d2, ue_b2, Re_b2, n2, ue_b1, Re_b1, n1,
                             BT, BIC_2, BIC_1, self.gal['disk_ell'], self.gal['pa']]
        
        self.decomp = bd_res


    def virgo_res(self, vals, plot=False):
        ang_r = np.arange(0, 180, 5)
        ang_r = np.append(ang_r, self.gal['pa'])
        virgo_res = np.zeros(len(ang_r))
        ue_d, Re_d, ue_b, Re_b, n = vals
        u0_d, h_d = self.back_transform(ue_d, Re_d, n=1)
        for i, theta_i in enumerate(ang_r):
            theta_ref = np.deg2rad(theta_i)
            x_data = self.target_angle(self.iso_data.T, theta_ref)
            model = self.combine(x_data, ue_b, u0_d, Re_b, h_d, n)
            virgo_res[i] = np.sum((model-self.mags)**2)/(len(x_data) - len(vals))
        
        if plot:
            theta_ref = np.deg2rad(ang_r[virgo_res.argmin()])
            x_data = self.target_angle(self.iso_data.T, theta_ref)
            fig, (ax1, ax2)= plt.subplots(figsize=(12, 8), dpi=100, nrows=2)
            x_ax = np.linspace(np.min(x_data), np.max(x_data), 100)

            ax1.plot(x_data, self.mags, 'k.', label=f'{ang_r[virgo_res.argmin()]} deg')
            ax1.plot(x_ax, self.combine(x_ax, ue_b, u0_d, Re_b, h_d, n), 'r-')
            ax1.set_xlabel('R [arcsec]', fontsize=12)
            ax1.set_ylabel('$\mu \:[mag\:arcsec^{-1}]$', fontsize=12)
            ax1.invert_yaxis()
            ax1.legend()

            ax2.plot(ang_r, virgo_res, 'b.')
            ax2.axvline(self.gal['pa'], c='k')
            ax2.set_xlabel(r'$\theta$')
            ax2.set_ylabel(r'$\chi^2$')
            plt.tight_layout()
            return fig, (ax1, ax2), virgo_res, ang_r

        return virgo_res


    def plot_BD(self, hor=1/np.zeros(6)):
        bd_res = self.decomp
        ang_r = np.arange(0, 180, 5)
        ang_r = np.append(ang_r, self.gal['pa'])
        cut = bd_res.T[-2]-bd_res.T[-1] < 0
        fig, ax = plt.subplots(figsize=(14, 6), ncols=3, nrows=2, dpi=100, sharex=True)
        ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()
        ax1.plot(ang_r[cut], bd_res.T[0][cut], 'b.')
        ax1.plot(ang_r[~cut], bd_res.T[0][~cut], 'r.')
        ax1.set_ylabel('Disk $\mu_e$')

        ax4.plot(ang_r[cut], bd_res.T[1][cut], 'b.')
        ax4.plot(ang_r[~cut], bd_res.T[1][~cut], 'r.')
        ax4.set_ylabel('Disk $R_e$')

        ax2.plot(ang_r[cut], bd_res.T[2][cut], 'b.')
        ax2.plot(ang_r[~cut], bd_res.T[2][~cut], 'r.')
        ax2.plot(ang_r, bd_res.T[5], 'g.', c='lime')
        ax2.set_ylabel('Bulge $\mu_e$')

        ax5.plot(ang_r[cut], bd_res.T[3][cut], 'b.')
        ax5.plot(ang_r[~cut], bd_res.T[3][~cut], 'r.')
        ax5.plot(ang_r, bd_res.T[6], 'g.', c='lime')
        ax5.set_ylabel('Bulge $R_e$')

        ax3.plot(ang_r[cut], bd_res.T[4][cut], 'b.')
        ax3.plot(ang_r[~cut], bd_res.T[4][~cut], 'r.')
        ax3.plot(ang_r, bd_res.T[7], 'g.', c='lime')
        ax3.set_ylabel('Sérsic n')

        ax6.plot(ang_r[cut], bd_res.T[8][cut], 'b.')
        ax6.plot(ang_r[~cut], bd_res.T[8][~cut], 'r.')
        ax6.set_ylabel('B/T')

        ax1.set_ylim([17, 27])
        ax2.set_ylim([17, 27])
        ax3.set_ylim([0, 6])
        ax4.set_ylim([0, len(self.image) * 0.262/2])
        ax5.set_ylim([0, len(self.image) * 0.262/2])
        ax6.set_ylim([0, 1])

        for ax_i in [ax4, ax5, ax6]:
            ax_i.set_xlabel(r'$\theta$')

        for hi in range(6):
            ax.flatten()[hi].axhline(hor[hi], c='k') 
            ax.flatten()[hi].axvline(self.gal['pa'], c='k')

        plt.tight_layout()
        return fig, ax