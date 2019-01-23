"""Search class"""

import os
import copy
import time
import pdb
import pickle

import numpy as np
import radvel
import radvel.fitting
from radvel.plot import orbit_plots

import rvsearch.periodogram as periodogram
import rvsearch.utils as utils


class Search:
    """Class to initialize and modify posteriors as planet search runs.

    Args:
        data (DataFrame): pandas dataframe containing times, vel, err, and insts.
        starname (str): String, used to name the output directory.
        max_planets (int): Integer, limit on iterative planet search.
        priors (list): List of radvel prior objects to use.
        crit (str): Either 'bic' or 'aic', depending on which criterion to use.
        fap (float): False-alarm-probability to pass to the periodogram object.
        min_per (float): Minimum search period, to pass to the periodogram object.
        dvdt (Boolean): Whether to include a linear trend in the search.
        curv (Boolean): Whether to include a quadratic trend in the search.
        fix (Boolean): Whether to fix known planet parameters during search.
        polish (Boolean): Whether to create finer period grid after planet is found.
        verbose (Boolean):

    """

    def __init__(self, data, post=None, starname='star', max_planets=8,
                priors=[], crit='bic', fap=0.01, min_per=3, manual_grid=None,
                trend=False, fix=False, polish=True, workers=1, verbose=True):

        if {'time', 'mnvel', 'errvel', 'tel'}.issubset(data.columns):
            self.data = data
            self.tels = np.unique(self.data['tel'].values)
        elif {'jd', 'mnvel', 'errvel', 'tel'}.issubset(data.columns):
            self.data = data
            self.tels = np.unique(self.data['tel'].values)
        else:
            raise ValueError('Incorrect data input.')

        self.starname = starname

        if post == None:
            self.priors = priors
            self.params = utils.initialize_default_pars(instnames=self.tels)
            self.post=utils.initialize_post(params=self.params,
                                            priors=self.priors)
        else:
            self.post = post
            self.priors = post.priors
        '''
        self.post = post
        self.params = self.post.params
        self.priors = self.post.priors
        '''
        self.all_params = []

        self.max_planets = max_planets
        if self.post.params.num_planets == 1 and self.post.params['k1'] == 0.:
            self.num_planets = 0
        else:
            self.num_planets = self.post.params.num_planets

        self.crit = crit
        '''
        # Play with calling __name__ of method
        if crit=='bic':
            self.crit = radvel.posterior.bic()
        eif crit=='aic':
            self.crit = radvel.posterior.aic()
        self.critname = self.crit.__string__
        else:
            raise ValueError('Invalid information criterion.')
        '''
        self.fap = fap
        self.min_per = min_per

        self.trend = trend
        self.fix = fix
        self.polish = polish

        self.manual_grid = manual_grid
        self.workers = workers
        self.verbose = verbose

        self.basebic = None

        self.pers = None
        self.periodograms = []
        self.bic_threshes = []
        self.best_bics = []
    '''
    @classmethod
    def from_pandas(cls, data, priors=None):
        params = utils.initialize_default_pars(instnames=data.tel)
        post = utils.initialize_post(data, params=params, priors=priors)
        return cls(post)

    @classmethod
    def from_post(cls, post):
        return cls(post)
    '''
    def trend_test(self):
        # Perform 0-planet baseline fit.
        post1 = copy.deepcopy(self.post)
        post1.params['per1'].vary = False
        post1.params['k1'].vary = False
        post1 =radvel.fitting.maxlike_fitting(post1, verbose=False)

        trend_curve_bic = self.post.likelihood.bic()

        # Test without curvature
        post1.params['curv'].value = 0.0
        post1.params['curv'].vary = False
        post1 = radvel.fitting.maxlike_fitting(post1, verbose=False)
        post1.params['dvdt'].vary = False

        trend_bic = post1.likelihood.bic()

        # Test without trend or curvature
        post2 = copy.deepcopy(post1)

        post2.params['dvdt'].value = 0.0
        post2.params['dvdt'].vary = False
        post2 = radvel.fitting.maxlike_fitting(post2, verbose=False)
        post1.params['curv'].vary = False

        flat_bic = post2.likelihood.bic()

        if trend_bic < flat_bic - 10.:
            if trend_curve_bic < trend_bic - 10.:
                # Quadratic
                pass
            else:
                # Linear
                self.post.params['dvdt'].value = post1.params['dvdt'].value
                self.post.params['curv'].value = 0
                self.post.params['curv'].vary = False
        else:
            # Flat
            self.post.params['dvdt'].value = 0
            self.post.params['dvdt'].vary = False
            self.post.params['curv'].value = 0
            self.post.params['curv'].vary = False


    def add_planet(self):

        current_num_planets = self.post.params.num_planets
        fitting_basis = self.post.params.basis.name
        param_list = fitting_basis.split()

        new_num_planets = current_num_planets + 1

        default_pars = utils.initialize_default_pars(instnames=self.tels)
        new_params = radvel.Parameters(new_num_planets, basis=fitting_basis)

        for planet in np.arange(1, new_num_planets):
            for par in param_list:
                parkey = par + str(planet)
                new_params[parkey] = self.post.params[parkey]

        for par in self.post.likelihood.extra_params:
            new_params[par] = self.post.params[par]  # For gamma and jitter

        # Set default parameters for n+1th planet
        default_params = utils.initialize_default_pars(self.tels)
        for par in param_list:
            parkey = par + str(new_num_planets)
            onepar = par + '1'  # MESSY, FIX THIS 10/22/18
            new_params[parkey] = default_params[onepar]

        new_params['dvdt'] = self.post.params['dvdt']
        new_params['curv'] = self.post.params['curv']

        if not self.post.params['dvdt'].vary:
            new_params['dvdt'].vary = False
        if not self.post.params['curv'].vary:
            new_params['curv'].vary = False

        new_params['per{}'.format(new_num_planets)].vary = False
        new_params['secosw{}'.format(new_num_planets)].vary = False
        new_params['sesinw{}'.format(new_num_planets)].vary = False

        new_params.num_planets = new_num_planets

        # TO-DO: Figure out how to handle jitter prior, whether needed
        if self.priors is not None:
            new_post = utils.initialize_post(self.data, new_params,
                                                        self.priors)
        else:
            priors = []
            priors.append(radvel.prior.PositiveKPrior(new_num_planets))
            priors.append(radvel.prior.EccentricityPrior(new_num_planets))
            new_post = utils.initialize_post(self.data, new_params, priors)
        self.post = new_post


    def sub_planet(self):

        current_num_planets = self.post.params.num_planets
        fitting_basis = self.post.params.basis.name
        param_list = fitting_basis.split()

        new_num_planets = current_num_planets - 1

        default_pars = utils.initialize_default_pars(instnames=self.tels)
        new_params = radvel.Parameters(new_num_planets, basis=fitting_basis)

        for planet in np.arange(1, new_num_planets+1):
            for par in param_list:
                parkey = par + str(planet)
                new_params[parkey] = self.post.params[parkey]

        # Add gamma and jitter params to the dictionary.
        for par in self.post.likelihood.extra_params:
            new_params[par] = self.post.params[par]

        new_params['dvdt'] = self.post.params['dvdt']
        new_params['curv'] = self.post.params['curv']

        if not self.post.params['dvdt'].vary:
            new_params['dvdt'].vary = False
        if not self.post.params['curv'].vary:
            new_params['curv'].vary = False

        priors = []
        priors.append(radvel.prior.PositiveKPrior(new_num_planets))
        priors.append(radvel.prior.EccentricityPrior(new_num_planets))

        new_post = utils.initialize_post(self.data, new_params, priors)
        self.post = new_post


    def fit_orbit(self):
        for planet in np.arange(1, self.num_planets+1):
            self.post.params['per{}'.format(planet)].vary = True
            self.post.params['k{}'.format(self.num_planets)].vary = True
            self.post.params['tc{}'.format(self.num_planets)].vary = True
            self.post.params['secosw{}'.format(self.num_planets)].vary = True
            self.post.params['sesinw{}'.format(self.num_planets)].vary = True

        if self.polish:
            # Make a finer, narrow period grid, and search with eccentricity.
            self.post.params['per{}'.format(self.num_planets)].vary = False
            default_pdict = {}
            for k in self.post.params.keys():
                default_pdict[k] = self.post.params[k].value
            polish_params = []
            polish_bics = []
            peak = np.argmax(self.periodograms[-1])
            if peak == len(self.periodograms[-1]) - 1:
                subgrid = np.linspace(self.pers[peak-1], 2*self.pers[peak] -
                                                        self.pers[peak-1], 9)
            else: #TO-DO: JUSTIFY 9 GRID POINTS, OR TAKE AS ARGUMENT
                subgrid = np.linspace(self.pers[peak-1], self.pers[peak+1], 9)
            fit_params = []
            power = []

            for per in subgrid:
                for k in default_pdict.keys():
                    self.post.params[k].value = default_pdict[k]
                perkey = 'per{}'.format(self.num_planets)
                self.post.params[perkey].value = per

                fit = radvel.fitting.maxlike_fitting(self.post, verbose=False)
                power.append(-fit.likelihood.bic())

                best_params = {}
                for k in fit.params.keys():
                    best_params[k] = fit.params[k].value
                fit_params.append(best_params)

            fit_index = np.argmax(power)
            bestfit_params = fit_params[fit_index]
            for k in self.post.params.keys():
                self.post.params[k].value = bestfit_params[k]
            self.post.params['per{}'.format(self.num_planets)].vary = True

        self.post = radvel.fitting.maxlike_fitting(self.post, verbose=False)
        if self.fix:
            for planet in np.arange(1, self.num_planets+1):
                self.post.params['per{}'.format(planet)].vary = False
                self.post.params['k{}'.format(self.num_planets)].vary = False
                self.post.params['tc{}'.format(self.num_planets)].vary = False
                self.post.params['secosw{}'.format(self.num_planets)].vary \
                                                                    = False
                self.post.params['sesinw{}'.format(self.num_planets)].vary \
                                                                    = False

    def add_gp(self, inst=None):
        pass

    def sub_gp(self, num_gps=1):
        try:
            pass
        except:
            raise RuntimeError('Model contains fewer than {} Gaussian \
                                processes.'.format(num_gps))

    def save(self, filename=None):
        if filename is not None:
            self.post.writeto(filename)
        else:
            self.post.writeto('post_final.pkl')

    def run_search(self):
        # Use all of the above routines to run a search.
        outdir = os.path.join(os.getcwd(), self.starname)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if self.trend:
            self.trend_test()
        else:
            self.post.params['dvdt'].vary = False
            self.post.params['curv'].vary = False

        run = True
        while run:
            if self.num_planets != 0:
                self.add_planet()

            perioder = periodogram.Periodogram(self.post, basebic=self.basebic,
                                        minsearchp=self.min_per, fap=self.fap,
                                        manual_grid=self.manual_grid,
                                        workers=self.workers,
                                        verbose=self.verbose)
            t1 = time.process_time()

            perioder.per_bic()
            self.periodograms.append(perioder.power[self.crit])
            if self.num_planets == 0:
                self.pers = perioder.pers

            perioder.eFAP_thresh()
            self.bic_threshes.append(perioder.bic_thresh)
            self.best_bics.append(perioder.best_bic)
            perioder.plot_per()
            perioder.fig.savefig(outdir+'/dbic{}.pdf'.format(
                                        self.num_planets+1))

            t2 = time.process_time()
            if self.verbose:
                print('Time = {} seconds'.format(t2 - t1))

            if perioder.best_bic > perioder.bic_thresh:
                self.num_planets += 1
                for k in self.post.params.keys():
                    self.post.params[k].value = perioder.bestfit_params[k]
                self.fit_orbit()
                self.all_params.append(self.post.params)
                self.basebic = self.post.bic()
            else:
                self.sub_planet()
                run = False
            if self.num_planets >= self.max_planets:
                run = False
            # Generate an orbit plot.
            rvplot = orbit_plots.MultipanelPlot(self.post, saveplot=outdir+
                                '/orbit_plot{}.pdf'.format(self.num_planets))
            multiplot_fig, ax_list = rvplot.plot_multipanel()
            multiplot_fig.savefig(outdir+'/orbit_plot{}.pdf'.format(
                                                    self.num_planets))

        self.save(filename=outdir+'/post_final.pkl')
        pickle_out = open(outdir+'/search.pkl','wb')
        pickle.dump(self, pickle_out)
        pickle_out.close()

        periodograms_plus_pers = np.append([self.pers], self.periodograms,
                                                                axis=0).T
        threshs_and_pks = np.append([self.bic_threshes], [self.best_bics],
                                                                axis=0).T
        np.savetxt(outdir+'/pers_periodograms.csv', periodograms_plus_pers,
                                                header='period  BIC_array')
        np.savetxt(outdir+'/thresholds_and_peaks.csv', threshs_and_pks,
                                        header='threshold  best_bic')
