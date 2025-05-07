import math

def dms_to_dec(d, m, s):
    if d > 0:
        return abs(d) + abs(m / 60) + abs(s / 3600)
    else:
        return -(abs(d) + abs(m / 60) + abs(s / 3600))

def calculate_azimuth(coord1, coord2):
    """
    Calculate the azimuth between two geographical points.
    
    Parameters:
    coord1 (tuple): The latitude and longitude of the first point as a tuple
    coord2 (tuple): The latitude and longitude of the second point as a tuple
    
    Returns:
    float: The azimuth in degrees
    """
    
    # Extract the latitude and longitude of the two points
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate the change in longitude
    dlon = lon2 - lon1
    x = math.atan2(math.sin(dlon) * math.cos(lat2), math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    
    # Convert the azimuth from radians to degrees
    azimuth = math.degrees(x)
    
    # Normalize the azimuth to a value between 0 and 360 degrees
    azimuth = (azimuth + 360) % 360
    
    return azimuth