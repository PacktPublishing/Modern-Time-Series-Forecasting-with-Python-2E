import numpy as np


### TimeSynth package has an open issue on its autoregressive function.  
# As of 3/16/24 the issue is still open.
# Link to issue https://github.com/TimeSynth/TimeSynth/issues/23
# Thus below is the rebuilt function with fix.

class BaseSignal:
    """BaseSignal class

    Signature for all signal classes.

    """

    def __init__(self):
        raise NotImplementedError

    def sample_next(self, time, samples, errors):
        """Samples next point based on history of samples and errors

        Parameters
        ----------
        time : int
            time
        samples : array-like
            all samples taken so far
        errors : array-like
            all errors sampled so far

        Returns
        -------
        float
            sampled signal for time t

        """
        raise NotImplementedError

    def sample_vectorized(self, time_vector):
        """Samples for all time points in input

        Parameters
        ----------
        time_vector : array like
            all time stamps to be sampled
        
        Returns
        -------
        float
            sampled signal for time t

        """
        raise NotImplementedError
__all__ = ['AutoRegressive']


class AutoRegressive(BaseSignal):
    """Sample generator for autoregressive (AR) signals.
    
    Generates time series with an autogressive lag defined by the number of parameters in ar_param.
    NOTE: Only use this for regularly sampled signals
    
    Parameters
    ----------
    ar_param : list (default [None])
        Parameter of the AR(p) process
        [phi_1, phi_2, phi_3, .... phi_p]
    sigma : float (default 1.0)
        Standard deviation of the signal
    start_value : list (default [None])
        Starting value of the AR(p) process
        
    """
    
    def __init__(self, ar_param=[None], sigma=0.5, start_value=[None]):
        self.vectorizable = False
        ar_param.reverse()
        self.ar_param = ar_param
        self.sigma = sigma
        if start_value[0] is None:
            self.start_value = [0 for i in range(len(ar_param))]
        else:
            if len(start_value) != len(ar_param):
                raise ValueError("AR parameters do not match starting value")
            else:
                self.start_value = start_value
        self.previous_value = self.start_value

    def sample_next(self, time, samples, errors):
        """Sample a single time point

        Parameters
        ----------
        time : number
            Time at which a sample was required

        Returns
        -------
        ar_value : float
            sampled signal for time t
        """
        ar_value = sum(self.previous_value[i] * self.ar_param[i] for i in range(len(self.ar_param)))
        noise = np.random.normal(loc=0.0, scale=self.sigma)
        ar_value += noise  # ar_value is now a scalar, noise is a scalar too
        self.previous_value = self.previous_value[1:] + [ar_value]
        return ar_value

