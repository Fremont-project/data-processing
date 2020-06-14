import requests
from shapely.geometry import LineString, Point

def shortest_path_by_travel_time(start, end, stop_on_error=False):
    """
    Using Here API to get smooth path from start to end location
    where start and end are Point objects
    """
    here_url = 'https://route.ls.hereapi.com/routing/7.2/calculateroute.json?'
    api_key = '0tEyB2hXkoJNHxI00G5U2M_KIcioZ6Z6ZnI9_HtjAek'

    # convention on .shp files and kepler are lng, lat
    # Here API convention is lat, lng
    start_pos = 'street!!{},{}'.format(start.y, start.x)
    end_pos = 'geo!{},{}'.format(end.y, end.x)
    params = {'apiKey': api_key,
              'mode': 'fastest;car;traffic:disabled',
              'routeAttributes': 'shape',
              'waypoint0': start_pos,
              'waypoint1': end_pos}

    response = requests.get(here_url, params=params)
    if response.ok:
        body = response.json()
        route_shape = body['response']['route'][0]['shape']
        if route_shape:
            route = [[float(p) for p in list(point.split(','))[::-1]] for point in route_shape]
            route.insert(0, [start.x, start.y])
            route.append([end.x, end.y])
            return LineString(route)
        else:
            if stop_on_error:
                stop_code('No routes found for start={}, destination={}'.format(start, end))
            return None
    else:
        if stop_on_error:
            response.raise_for_status()
        return None 
    
def square(x): 
    return x * x