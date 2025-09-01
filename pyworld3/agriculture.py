# -*- coding: utf-8 -*-

# © Copyright Charles Vanwynsberghe (2021)

# Pyworld3 is a computer program whose purpose is to run configurable
# simulations of the World3 model as described in the book "Dynamics
# of Growth in a Finite World".

# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".

# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.

# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import os
import json

from scipy.interpolate import interp1d
import numpy as np
import inspect

from .specials import Smooth, clip
from .utils import requires, _create_control_function


class Agriculture:
    """
    Agriculture sector. Can be run independantly from other sectors with
    exogenous inputs. The initial code is defined p.362.

    Examples
    --------
    Running the agriculture sector alone requires artificial (exogenous) inputs
    which should be provided by the other sectors. Start from the following
    example:

    >>> agr = Agriculture()
    >>> agr.set_control_functions()
    >>> agr.set_agriculture_table_functions()
    >>> agr.init_agriculture_variables()
    >>> agr.init_agriculture_constants()
    >>> agr.set_agriculture_delay_functions()
    >>> agr.init_exogenous_inputs()
    >>> agr.run_agriculture()

    Parameters
    ----------
    year_min : float, optional
        start year of the simulation [year]. The default is 1900.
    year_max : float, optional
        end year of the simulation [year]. The default is 2100.
    dt : float, optional
        time step of the simulation [year]. The default is 0.25.
    verbose : bool, optional
        print information for debugging. The default is False.

    Attributes
    ----------
    ali : float, optional
        arable land initial [hectares]. The default is 0.9e9.
    pali : float, optional
        potentially arable land initial [hectares]. The default is 2.3e9.
    lfh : float, optional
        land fraction harvested []. The default is 0.7.
    palt : float, optional
        potentially arable land total [hectares]. The default is 3.2e9.
    pl : float, optional
        processing loss []. The default is 0.1.
    io70 : float, optional
        industrial output in 1970 [dollars/year]. The default is 7.9e11.
    sd : float, optional
        social discount [1/year]. The default is 0.07.
    uili : float, optional
        urban-industrial land initial [hectares]. The default is 8.2e6.
    alln : float, optional
        average life of land normal [years]. The default is 6000.
    uildt : float, optional
        urban-industrial land development time [years]. The default is 10.
    lferti : float, optional
        land fertility initial [vegetable-equivalent kilograms/hectare-year].
        The default is 600.
    ilf : float, optional
        inherent land fertility [vegetable-equivalent kilograms/hectare-year].
        The default is 600.
    fspd : float, optional
        food shortage perception delay [years]. The default is 2.
    sfpc : float, optional
        subsistence food per capita
        [vegetable-equivalent kilograms/person-year]. The default is 230.

    **Control signals**
    alai_control : function, optional
        alai, control function with argument time [years]. The default is 2.
    lyf_control : function, optional
        lyf, control function with argument time [years]. The default is 1.
    ifpc_control : function, optional
        fraction of normal ifpc used, control function with argument time [years]. The default is 1.0
    lymap_control : function, optional
        fraction of normal lymap used, control function with argument time [years]. The default is 1.0
    llmy_control : function, optional
        fraction of normal llmy used, control function with argument time [years]. The default is 1.0
    fioaa_control : function, optional
        fraction of normal fioaa used, control function with argument time [years]. The default is 1.0


    **Loop 1 - food from investment in land development**

    al : numpy.ndarray
        arable land [hectares].
    pal : numpy.ndarray
        potentially arable land [hectares].
    dcph : numpy.ndarray
        development cost per hectare [dollars/hectare].
    f : numpy.ndarray
        food [vegetable-equivalent kilograms/year].
    fpc : numpy.ndarray
        food per capita [vegetable-equivalent kilograms/person-year].
    fioaa : numpy.ndarray
        fraction of industrial output allocated to agriculture [].
    ifpc : numpy.ndarray
        indicated food per capita [vegetable-equivalent kilograms/person-year].
    ldr : numpy.ndarray
        land development rate [hectares/year].
    lfc : numpy.ndarray
        land fraction cultivated [].
    tai : numpy.ndarray
        total agricultural investment [dollars/year].

    **Loop 2 - food from investment in agricultural inputs**

    ai : numpy.ndarray
        agricultural inputs [dollars/year].
    aiph : numpy.ndarray
        agricultural inputs per hectare [dollars/hectare-year].
    alai : numpy.ndarray
        average lifetime of agricultural inputs [years].
    cai : numpy.ndarray
        current agricultural inputs [dollars/year].
    ly : numpy.ndarray
        land yield [vegetable-equivalent kilograms/hectare-year].
    lyf : numpy.ndarray
        land yield factor [].
    lymap : numpy.ndarray
        land yield multiplier from air pollution [].
    lymc : numpy.ndarray
        land yield multiplier from capital [].

    **Loop 1 & 2 - the investment allocation decision*

    fiald : numpy.ndarray
        fraction of inputs allocated to land development [].
    mlymc : numpy.ndarray
        marginal land yield multiplier from capital [hectares/dollar].
    mpai : numpy.ndarray
        marginal productivity of agricultural inputs
        [vegetable equivalent kilograms/dollar].
    mpld : numpy.ndarray
        marginal productivity of land development
        [vegetable-equivalent kilograms/dollar].

    **Loop 3 -land erosion and urban-industrial use**

    uil : numpy.ndarray
        urban-industrial land [hectares].
    all : numpy.ndarray
        average life of land [years].
    llmy : numpy.ndarray
        land life multiplier from yield [].
    ler : numpy.ndarray
        land erosion rate [hectares/year].
    lrui : numpy.ndarray
        land removal for urban-industrial use [hectares/year].
    uilpc : numpy.ndarray
        urban-industrial land per capita [hectares/person].
    uilr : numpy.ndarray
        urban-industrial land required [hectares].

    **Loop 4 - land fertility degradation**

    lfert : numpy.ndarray
        land fertility [vegetable-equivalent kilograms/hectare-year].
    lfd : numpy.ndarray
        land fertility degradation
        [vegetable-equivalent kilograms/hectare-year-year].
    lfdr : numpy.ndarray
        land fertility degradation rate [1/year].

    **Loop 5 - land fertility regeneration**

    lfr : numpy.ndarray
        land fertility regeneration
        [vegetable-equivalent kilograms/hectare-year-year].
    lfrt : numpy.ndarray
        land fertility regeneration time [years].

    **Loop 6 - discontinuing land maintenance**

    falm : numpy.ndarray
        fraction of inputs allocated to land maintenance [dimensionless].
    fr : numpy.ndarray
        food ratio [].
    pfr : numpy.ndarray
        perceived food ratio [].

    """

    def __init__(self, year_min=1900, year_max=2100, dt=0.25, verbose=False, prev_run_data=None):
        self.dt = dt
        self.year_min = year_min
        self.year_max = year_max
        self.verbose = verbose
        self.length = self.year_max - self.year_min
        self.n = int(self.length / self.dt) if prev_run_data is None else int(self.length / self.dt) + prev_run_data['n']
        self.time = np.arange(self.year_min, self.year_max + self.dt, self.dt) if prev_run_data is None else np.concatenate(prev_run_data['time'],np.arange(self.year_min, self.year_max + self.dt, self.dt))
        

    def set_agriculture_control(self, **control_functions):
        """
        Define the control commands. Their units are documented above at the class level.
        """
        default_control_functions = {
            "alai_control": lambda _: 2,
            "lyf_control": lambda _: 1,
            "ifpc_control": lambda _: 1,
            "lymap_control": lambda _: 1,
            "llmy_control": lambda _: 1,
            "fioaa_control": lambda _: 1,
        }
        _create_control_function(self, default_control_functions, control_functions)

    def init_agriculture_constants(
        self,
        ali=0.9e9,
        pali=2.3e9,
        lfh=0.7,
        palt=3.2e9,
        pl=0.1,
        io70=7.9e11,
        sd=0.07,
        uili=8.2e6,
        alln=6000,
        uildt=10,
        lferti=600,
        ilf=600,
        fspd=2,
        sfpc=230,
    ):
        """
        Initialize the constant parameters of the agriculture sector.
        Constants and their unit are documented above at the class level.

        """
        # loop 1 - food from investment in land development
        self.ali = ali
        self.pali = pali
        self.lfh = lfh
        self.palt = palt
        self.pl = pl
        # loop 2 - food from investment in agricultural inputs
        self.io70 = io70
        # loop 1 & 2 - the investment allocation decision
        self.sd = sd
        # loop 3 -land erosion and urban-industrial use
        self.uili = uili
        self.alln = alln
        self.uildt = uildt
        # loop 4 - land fertility degradation
        self.lferti = lferti
        # loop 5 - land fertility regeneration
        self.ilf = ilf
        # loop 6 - discontinuing land maintenance
        self.fspd = fspd
        self.sfpc = sfpc

    def init_agriculture_variables(self, prev_run_data={}):
        """
        Initialize the state and rate variables of the agriculture sector
        with the option to use data from a previous run.

        :param prev_run_data: A dictionary containing arrays from a previous run, keyed by variable name.
        """
        # List of agriculture sector variables
        variables = [
            "al", "pal", "dcph", "f", "fpc", "fioaa", "ifpc", "ldr", "lfc", "tai",
            "ai", "aiph", "alai", "cai", "ly", "lyf", "lymap", "lymc",
            "fiald", "mlymc", "mpai", "mpld",
            "uil", "all", "llmy", "ler", "lrui", "uilpc", "uilr",
            "lfert", "lfd", "lfdr",
            "lfr", "lfrt",
            "falm", "fr", "pfr"
        ]
        
        # Initialize variables either with previous run data or as new arrays
        for var in variables:
            if bool(prev_run_data):
                for var in prev_run_data:
                    # Get the array from prev_run_data
                    original_array = prev_run_data[var]
                    nan_extension_size = self.n - len(original_array)
                    if nan_extension_size > 0:
                        # Extend the array with nan values if needed
                        extended_array = np.concatenate([original_array, np.full(nan_extension_size, np.nan)])
                        setattr(self, var, extended_array)
                    else:
                        # If the original array is already the correct size or larger, just use it as is
                        setattr(self, var, original_array)
            else:
                setattr(self, var, np.full((self.n,), np.nan))

    def set_agriculture_delay_functions(self, method="euler", prev_run_data = {}):
        """
        Set the linear smoothing and delay functions for the agriculture sector,
        potentially using data from a previous run.

        :param method: Numerical integration method: "euler" or "odeint".
        :param prev_run_data: Optional. Data from a previous run to ensure continuity in delay functions.
        """
        var_smooth = ["CAI", "FR"]
        for var_ in var_smooth:
            data = getattr(self, var_.lower())
            func_delay = Smooth(data, self.dt, self.time, method=method)
            if bool(prev_run_data):
                original_out_arr = prev_run_data['smooth_' + var_.lower()]
                for i in range(len(original_out_arr)):
                    func_delay.out_arr[i] = original_out_arr[i]

            setattr(self, "smooth_" + var_.lower(), func_delay)

    def set_agriculture_table_functions(self, json_file=None):
        """
        Set the nonlinear functions of the agriculture sector, based on a json
        file. By default, the `functions_table_world3.json` file from pyworld3
        is used.

        Parameters
        ----------
        json_file : file, optional
            json file containing all tables. The default is None.

        """
        if json_file is None:
            json_file = "./functions_table_world3.json"
            json_file = os.path.join(os.path.dirname(__file__), json_file)
        with open(json_file) as fjson:
            tables = json.load(fjson)

        func_names = [
            "IFPC",
            "FIOAA",
            "DCPH",
            "LYMC",
            "LYMAP",
            "FIALD",
            "MLYMC",
            "LLMY",
            "UILPC",
            "LFDR",
            "LFRT",
            "FALM",
        ]

        for func_name in func_names:
            for table in tables:
                if table["y.name"] == func_name:
                    func = interp1d(
                        table["x.values"],
                        table["y.values"],
                        bounds_error=False,
                        fill_value=(table["y.values"][0], table["y.values"][-1]),
                    )
                    setattr(self, func_name.lower() + "_f", func)

    def init_exogenous_inputs(self):
        """
        Initialize all the necessary constants and variables to run the
        agriculture sector alone. These exogenous parameters are outputs from
        the 4 other remaining sectors in a full simulation of World3.

        """
        # constants
        self.eyear = 2500
        self.popi = 1.65e9
        self.ioi = 0.67e11
        self.ppolxi = 0.12
        # variables
        self.pop = np.zeros((self.n,))
        self.pop1 = np.zeros((self.n,))
        self.pop2 = np.zeros((self.n,))
        self.io = np.zeros((self.n,))
        self.io1 = np.zeros((self.n,))
        self.io2 = np.zeros((self.n,))
        self.iopc = np.zeros((self.n,))
        self.ppolx = np.zeros((self.n,))
        self.ppolx1 = np.zeros((self.n,))
        self.ppolx2 = np.zeros((self.n,))

    @requires(["pop", "io", "iopc", "ppolx"])
    def loopk_exogenous(self, k):
        """
        Run a sorted sequence to update one loop of the exogenous parameters.
        `@requires` decorator checks that all dependencies are computed
        previously.

        """
        self.pop1[k] = self.popi * np.exp(0.012 * (self.time[k] - self.year_min))
        self.pop2[k] = self.popi * np.exp(0.012 * (self.eyear - self.year_min))
        self.pop[k] = clip(self.pop2[k], self.pop1[k], self.time[k], self.eyear)

        self.io1[k] = self.ioi * np.exp(0.036 * (self.time[k] - self.year_min))
        self.io2[k] = self.ioi * np.exp(0.036 * (self.eyear - self.year_min))
        self.io[k] = clip(self.io2[k], self.io1[k], self.time[k], self.eyear)
        self.iopc[k] = self.io[k] / self.pop[k]

        self.ppolx1[k] = self.ppolxi * np.exp(0.03 * (self.time[k] - self.year_min))
        self.ppolx2[k] = self.ppolxi * np.exp(0.03 * (self.eyear - self.year_min))
        self.ppolx[k] = clip(self.ppolx2[k], self.ppolx1[k], self.time[k], self.eyear)

    def loop0_exogenous(self):
        """
        Run a sequence to initialize the exogenous parameters (loop with k=0).

        """
        self.loopk_exogenous(0)

    def loop0_agriculture(self, alone=False):
        """
        Run a sequence to initialize the population sector (loop with k=0).

        Parameters
        ----------
        alone : boolean, optional
            if True, run the sector alone with exogenous inputs. The default
            is False.

        """
        # Set initial conditions
        self.al[0] = self.ali
        self.pal[0] = self.pali
        self.uil[0] = self.uili
        self.lfert[0] = self.lferti
        self.ai[0] = 5e9
        self.pfr[0] = 1
        if alone:
            self.loop0_exogenous()
        self._update_lfc(0)
        self._update_f(0)
        self._update_fpc(0)
        self._update_ifpc(0)
        self._update_fioaa(0)
        self._update_tai(0)
        self._update_dcph(0)
        # loop 1&2
        self._update_mlymc(0)
        self._update_mpai(0)
        self._update_mpld(0)
        self._update_fiald(0)
        # back to loop 1
        self._update_ldr(0, 0)
        # loop 2
        self._update_cai(0)
        self._update_alai(0)
        # loop 6
        self._update_falm(0)
        self._update_fr(0)
        # back to loop 2
        self._update_aiph(0)
        self._update_lymc(0)
        self._update_lyf(0)
        self._update_lymap(0)
        # loop 4
        self._update_lfdr(0)
        # back to loop 2
        self._update_lfd(0, 0)
        self._update_ly(0)
        # loop 3
        self._update_all(0)
        self._update_llmy(0)
        self._update_ler(0, 0)
        self._update_uilpc(0)
        self._update_uilr(0)
        self._update_lrui(0, 0)
        # loop 5
        self._update_lfr(0, 0)
        self._update_lfrt(0)
        # recompute supplementary initial conditions
        self._update_ai(0)
        self._update_pfr(0)

    def loopk_agriculture(self, j, k, jk, kl, alone=False):
        """
        Run a sequence to update one loop of the agriculture sector.

        Parameters
        ----------
        alone : boolean, optional
            if True, run the sector alone with exogenous inputs. The default
            is False.

        """
        redo = self.redo_loop
        if alone:
            self.loopk_exogenous(k)
        # Update state variables
        self._update_state_al(k, j, jk)
        self._update_state_pal(k, j, jk)
        self._update_state_uil(k, j, jk)
        self._update_state_lfert(k, j, jk)
        # loop 1
        self._update_lfc(k)
        self._update_f(k)
        self._update_fpc(k)
        self._update_ifpc(k)
        self._update_fioaa(k)
        self._update_tai(k)
        self._update_dcph(k)
        # loop 1&2
        self._update_mlymc(k)
        self._update_mpai(k)
        self._update_mpld(k)
        self._update_fiald(k)
        # back to loop 1
        self._update_ldr(k, kl)
        # loop 2
        self._update_cai(k)
        self._update_alai(k)
        self._update_ai(k)  # !!! checks cai for all k but useless if >=1
        # loop 6
        self._update_pfr(k)
        self._update_falm(k)
        self._update_fr(k)
        # back to loop 2
        self._update_aiph(k)
        self._update_lymc(k)
        self._update_lyf(k)
        self._update_lymap(k)
        # loop 4
        self._update_lfdr(k)
        # back to loop 2
        self._update_lfd(k, kl)
        self._update_ly(k)
        # loop 3
        self._update_all(k)
        self._update_llmy(k)
        self._update_ler(k, kl)
        self._update_uilpc(k)
        self._update_uilr(k)
        self._update_lrui(k, kl)
        # loop 5
        self._update_lfr(k, kl)
        self._update_lfrt(k)

    def run_agriculture(self):
        """
        Run a sequence of updates to simulate the agriculture sector alone with
        exogenous inputs.

        """
        self.redo_loop = True
        while self.redo_loop:
            self.redo_loop = False
            self.loop0_agriculture(alone=True)

        for k_ in range(1, self.n):
            self.redo_loop = True
            while self.redo_loop:
                self.redo_loop = False
                if self.verbose:
                    print("go loop", k_)
                self.loopk_agriculture(k_ - 1, k_, k_ - 1, k_, alone=True)

    @requires(["lfc"], ["al"])
    def _update_lfc(self, k):
        """
        From step k requires: AL
        """
        self.lfc[k] = self.al[k] / self.palt

    @requires(["al"])
    def _update_state_al(self, k, j, jk):
        """
        State variable, requires previous step only
        """
        self.al[k] = self.al[j] + self.dt * (
            self.ldr[jk] - self.ler[jk] - self.lrui[jk]
        )

    @requires(["pal"])
    def _update_state_pal(self, k, j, jk):
        """
        State variable, requires previous step only
        """
        self.pal[k] = self.pal[j] - self.dt * self.ldr[jk]

    @requires(["f"], ["ly", "al"])
    def _update_f(self, k):
        """
        From step k requires: LY AL
        """
        self.f[k] = self.ly[k] * self.al[k] * self.lfh * (1 - self.pl)
        
        

    @requires(["fpc"], ["f", "pop"])
    def _update_fpc(self, k):
        """
        From step k requires: F POP
        """
        self.fpc[k] = self.f[k] / self.pop[k]
        


    @requires(["ifpc"], ["iopc"])
    def _update_ifpc(self, k):
        """
        From step k requires: IOPC
        """
        self.ifpc[k] = self.ifpc_control* self.ifpc_f(self.iopc[k])

    @requires(["tai"], ["io", "fioaa"])
    def _update_tai(self, k):
        """
        From step k requires: IO FIOAA
        """
        self.tai[k] = self.io[k] * self.fioaa[k]

    @requires(["fioaa"], ["fpc", "ifpc"])
    def _update_fioaa(self, k):
        """
        From step k requires: FPC IFPC
        """
        self.fioaa[k] = self.fioaa_control * self.fioaa_f(
            self.fpc[k] / self.ifpc[k]
        )

    @requires(["ldr"], ["tai", "fiald", "dcph"])
    def _update_ldr(self, k, kl):
        """
        From step k requires: TAI FIALD DCPH
        """
        self.ldr[kl] = self.tai[k] * self.fiald[k] / self.dcph[k]

    @requires(["dcph"], ["pal"])
    def _update_dcph(self, k):
        """
        From step k requires: PAL
        """
        self.dcph[k] = self.dcph_f(self.pal[k] / self.palt)

    @requires(["cai"], ["tai", "fiald"])
    def _update_cai(self, k):
        """
        From step k requires: TAI FIALD
        """
        self.cai[k] = self.tai[k] * (1 - self.fiald[k])

    # OPTIMIZE checks more than necessary (cai[k] for k>=1)
    @requires(["ai"], ["cai", "alai"])
    def _update_ai(self, k):
        """
        From step k=0 requires: CAI, else nothing
        """
        self.ai[k] = self.smooth_cai(k, self.alai[k])

    @requires(["alai"])
    def _update_alai(self, k):
        """
        From step k requires: nothing
        """
        self.alai[k] = self.alai_control

    @requires(["aiph"], ["ai", "falm", "al"])
    def _update_aiph(self, k):
        """
        From step k requires: AI FALM AL
        """
        self.aiph[k] = self.ai[k] * (1 - self.falm[k]) / self.al[k]

    @requires(["lymc"], ["aiph"])
    def _update_lymc(self, k):
        """
        From step k requires: AIPH
        """
        self.lymc[k] = self.lymc_f(self.aiph[k])

    @requires(["ly"], ["lyf", "lfert", "lymc", "lymap"])
    def _update_ly(self, k):
        """
        From step k requires: LYF LFERT LYMC LYMAP
        """
        self.ly[k] = self.lyf[k] * self.lfert[k] * self.lymc[k] * self.lymap[k]
        

    @requires(["lyf"])
    def _update_lyf(self, k):
        """
        From step k requires: nothing
        """
        self.lyf[k] = self.lyf_control

    @requires(["lymap"], ["io"])
    def _update_lymap(self, k):
        """
        From step k requires: IO
        """
        self.lymap[k] = self.lymap_control * self.lymap_f(
            self.io[k] / self.io70
        )
        

    @requires(["fiald"], ["mpld", "mpai"])
    def _update_fiald(self, k):
        """
        From step k requires: MPLD MPAI
        """
        self.fiald[k] = self.fiald_f(self.mpld[k] / self.mpai[k])

    @requires(["mpld"], ["ly", "dcph"])
    def _update_mpld(self, k):
        """
        From step k requires: LY DCPH
        """
        self.mpld[k] = self.ly[k] / (self.dcph[k] * self.sd)

    @requires(["mpai"], ["alai", "ly", "mlymc", "lymc"])
    def _update_mpai(self, k):
        """
        From step k requires: ALAI LY MLYMC LYMC
        """
        self.mpai[k] = self.alai[k] * self.ly[k] * self.mlymc[k] / self.lymc[k]

    @requires(["mlymc"], ["aiph"])
    def _update_mlymc(self, k):
        """
        From step k requires: AIPH
        """
        self.mlymc[k] = self.mlymc_f(self.aiph[k])

    @requires(["all"], ["llmy"])
    def _update_all(self, k):
        """
        From step k requires: LLMY
        """
        self.all[k] = self.alln * self.llmy[k]

    @requires(["llmy"], ["ly"])
    def _update_llmy(self, k):
        """
        From step k requires: LY
        """
        self.llmy[k] = self.llmy_control* self.llmy_f(self.ly[k] / self.ilf)

    @requires(["ler"], ["al", "all"])
    def _update_ler(self, k, kl):
        """
        From step k requires: AL ALL
        """
        self.ler[kl] = self.al[k] / self.all[k]

    @requires(["uilpc"], ["iopc"])
    def _update_uilpc(self, k):
        """
        From step k requires: IOPC
        """
        self.uilpc[k] = self.uilpc_f(self.iopc[k])

    @requires(["uilr"], ["uilpc", "pop"])
    def _update_uilr(self, k):
        """
        From step k requires: UILPC POP
        """
        self.uilr[k] = self.uilpc[k] * self.pop[k]

    @requires(["lrui"], ["uilr", "uil"])
    def _update_lrui(self, k, kl):
        """
        From step k requires: UILR UIL
        """
        self.lrui[kl] = np.maximum(0, (self.uilr[k] - self.uil[k]) / self.uildt)

    @requires(["uil"])
    def _update_state_uil(self, k, j, jk):
        """
        State variable, requires previous step only
        """
        self.uil[k] = self.uil[j] + self.dt * self.lrui[jk]

    @requires(["lfert"])
    def _update_state_lfert(self, k, j, jk):
        """
        State variable, requires previous step only
        """
        self.lfert[k] = self.lfert[j] + self.dt * (self.lfr[jk] - self.lfd[jk])

    @requires(["lfdr"], ["ppolx"])
    def _update_lfdr(self, k):
        """
        From step k requires: PPOLX
        """
        self.lfdr[k] = self.lfdr_f(self.ppolx[k])

    @requires(["lfd"], ["lfert", "lfdr"])
    def _update_lfd(self, k, kl):
        """
        From step k requires: LFERT LFDR
        """
        self.lfd[kl] = self.lfert[k] * self.lfdr[k]

    @requires(["lfr"], ["lfert", "lfrt"])
    def _update_lfr(self, k, kl):
        """
        From step k requires: LFERT LFRT
        """
        self.lfr[kl] = (self.ilf - self.lfert[k]) / self.lfrt[k]

    @requires(["lfrt"], ["falm"])
    def _update_lfrt(self, k):
        """
        From step k requires: FALM
        """
        self.lfrt[k] = self.lfrt_f(self.falm[k])

    @requires(["falm"], ["pfr"])
    def _update_falm(self, k):
        """
        From step k requires: PFR
        """
        self.falm[k] = self.falm_f(self.pfr[k])

    @requires(["fr"], ["fpc"])
    def _update_fr(self, k):
        """
        From step k requires: FPC
        """
        self.fr[k] = self.fpc[k] / self.sfpc

    @requires(["pfr"], ["fr"], check_after_init=False)
    def _update_pfr(self, k):
        """
        From step k=0 requires: FR, else nothing
        """
        self.pfr[k] = self.smooth_fr(k, self.fspd)
