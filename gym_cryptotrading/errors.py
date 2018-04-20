from gym.error import Error

class EnvironmentAlreadyLoaded(Error):
    '''
    Raised when user tries to set the parameters of the environment that is
    already loaded.
    '''
    pass
