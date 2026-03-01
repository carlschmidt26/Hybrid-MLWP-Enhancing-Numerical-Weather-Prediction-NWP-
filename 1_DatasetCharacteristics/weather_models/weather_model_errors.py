### Exceptions ###
class WeatherModelException(Exception):
    """Base exception class for all exceptions related to the WeatherModelBase class."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    # The output of `__str__()` is printed as the last part ('detail') of the message.
    def __str__(self):
        return self.message

class DownloadFailedException(WeatherModelException):
    """Exception raised when the download for an existing file failed."""
    pass
    
class DynamicDimensionsCoordinateRenamingException(WeatherModelException):
    """Exception raised when the dynamic renaming of an Xarray Dimensions Coordinate failed."""
    pass

class DataProccesingException(WeatherModelException):
    """Exception raised when the dynamic renaming of an Xarray Dimensions Coordinate failed."""
    pass
    
### Warnings ###
class WeatherModelWarning(Warning):
    """Base warning class for all exceptions related to the WeatherModelBase class."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    # The output of `__str__()` is printed as the last part ('detail') of the message.
    def __str__(self):
        return self.message

class RemoteFileNotFoundWarning(WeatherModelWarning):
    """Warning raised in case the file does not exist on the server."""
    pass
    
class CorruptedFileWarning(WeatherModelWarning):
    """Warning raised in case the file does not exist on the server."""
    pass