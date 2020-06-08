# Credits: Code adapted and heavily modified from that of Attila Nagy, PhD Student at Budapest University of Technology and Economics (https://gitlab.medianets.hu/anagy)

import os
import io
import abc
import math
import gzip
import getpass
import logging
import zipfile
import requests
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from shapely import geometry
from typing import Any, Generator, Optional
from datetime import date, timedelta, datetime

logger = logging.getLogger(__name__)

_IMPORTED_FOLDER = ".imported"


def get_imported_folder_path():
    return _IMPORTED_FOLDER


def set_imported_folder_path(new_path):
    global _IMPORTED_FOLDER
    _IMPORTED_FOLDER = new_path


def date_range(date1, date2) -> Generator[date, Any, None]:
    for n in range(int((date2 - date1).days)+1):
        yield (date1 + timedelta(n))


def filename_to_date_1(filename: str):
    parts = filename.split('.')[0].split('_')[-3:]
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def filename_to_date_2(filename: str):
    parts = filename.split('.')[0].split('_')[-3:]
    return date(int(parts[1]), int(parts[2]), 1)


def get_caltrans_districts(caltrans_shapefile):
    return gpd.read_file(caltrans_shapefile)


def fetch_pems_from_ctdistrict(ctdistrict, start_date, end_date, mode):   
    if mode == "raw":
        data_handler = StationRawDataHandler()
    elif mode == "meta":
        data_handler = StationMetaDataHandler()
    elif mode == "5min":
        data_handler = Station5MinDataHandler()
    elif mode == "hour":
        data_handler = StationHourDataHandler()
    
    print("Downloading {} data...".format(mode))
    
    data_chunks = data_handler.load_between(from_date=start_date, to_date=end_date, district=ctdistrict)
    
    return pd.concat([data for data in data_chunks])


def fetch_pems_from_ctdistricts(ctdistricts, start_date, end_date, mode):
    if len(ctdistricts) == 1:
        return fetch_pems_from_ctdistrict(ctdistricts[0], start_date, end_date, mode)
    
    ctdistricts_data = []
    
    for ctdistrict in ctdistricts:
        ctdistrict_data = fetch_pems_from_ctdistrict(ctdistrict, start_date, end_date)
        ctdistricts_data.append(ctdistrict_data)
        
    return pd.concat(ctdistricts_data)


def fetch_pems_from_polygon(polygon, start_date, end_date, caltrans_shapefile, mode):
    caltrans_districts = get_caltrans_districts(caltrans_shapefile)
    
    polygon_districts = list(caltrans_districts[caltrans_districts['geometry'].contains(polygon)]['DISTRICT'].values)
    
    polygon_districts_data = fetch_pems_from_ctdistricts(polygon_districts, start_date, end_date, mode)
    
    if mode == "meta":
        polygon_districts_data = polygon_districts_data.rename(columns={'id': 'station_id'})
    
    polygon_districts_metadata = fetch_pems_from_ctdistricts(polygon_districts, start_date, end_date, "meta")
    polygon_districts_metadata = polygon_districts_metadata.rename(columns={'id': 'station_id'})
    polygon_districts_metadata = gpd.GeoDataFrame(polygon_districts_metadata,\
                                                  geometry=gpd.points_from_xy(polygon_districts_metadata['longitude'],\
                                                                              polygon_districts_metadata['latitude']))
    
    print("Cleaning data...")
    polygon_districts_data_merged = polygon_districts_metadata.merge(polygon_districts_data, on = ['station_id'])
    del polygon_districts_data, polygon_districts_metadata
    
    print("Filtering data...")
    polygon_data = polygon_districts_data_merged[polygon_districts_data_merged['geometry'].within(polygon)].reset_index()
    
    print("Done! Caltrans PeMS data for input region in specified date range has been generated and returned as a DataFrame.")
    
    return polygon_data


def fetch_pems_from_city(city_name, start_date, end_date, caltrans_shapefile, mode):
    city_gdf = ox.gdf_from_place(city_name)
    city_polygon = city_gdf['geometry'].values[0]
    
    return fetch_pems_from_polygon(city_polygon, start_date, end_date, caltrans_shapefile, mode)
    
    
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PeMSConnection(object, metaclass=Singleton):
    """
    The singleton class is responsible for handling data download through the official PeMS site. It never caches the
    downloaded data files.

    Warning note:
    Note that pems is not using HTTPS secure connection. Any credentials you will provide will go through the internet
    unencrypted and can be potentially see by someone who should not see them.

    Before any download operation, the connection must be initialized by initialize() function.

    """

    SITE_URL = "http://pems.dot.ca.gov/"

    def __init__(self):
        self._username = None
        self.session: requests.Session = None
        self._chunk_urls = {}  # url cache for repeated request

    @staticmethod
    def initialize():
        username = getpass.getpass(prompt='PeMS Username: ')
        password = getpass.getpass(prompt='PeMS Password: ')
        
        PeMSConnection()._create_new_session(username=username, password=password)

    @property
    def initialized(self) -> bool:
        """
        True if the session variable was initialized, otherwise it is false.

        :return:
        """
        return not self.session is None

    def destroy(self):
        """
        Destroy the current session and forget the username too.

        :return:
        """
        self.session = None
        self._username = None

    def _create_new_session(self, username: str, password: str):
        """
        Creates a new session using the input credentials.

        :param username:
        :param password:
        :return:
        """
        # Create new session
        self.session = requests.Session()
        # Generate site cookie
        self.session.get(self.SITE_URL)
        # Initiate login process
        response = self.session.post(self.SITE_URL, data={
            'username': username,
            'password': password,
            'login': 'Login'
        })
        # Check response
        if 'Incorrect username or password' in response.text:
            self.session = None
            logger.error("Incorrect username or password. Please check the provided credentials.")
        else:
            self._username = username
            logger.info("PeMSConnection has been successfully initialized.")
            
            print("\nA connection has been successfully initialized! Please note that the PeMS website is not secure.\n")

    def download(self, url, file_path=None):
        """
        Downloads the file specified by the url parameter and returns with a BytesIO buffer which stores the content
        of the file.

        If the file_path is set, the content of the buffer will be saved to the file_path.

        :param url:
        :param file_path:
        :return:
        """


        # If the connection is not initialized, the file download is not available.
        if not self.initialized:
            raise RuntimeError("PeMSConnection has been not initialized yet. Use PeMSConnection.initialize() "
                               "for this purpose.")

        if url is None:
            return None

        # Send HTTP GET request using the url parameter
        r = self.session.get(url, stream=True)

        # At first, the file will be downloaded in a temporary byte buffer chunk by chunk
        buffer = io.BytesIO()

        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        wrote = 0
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB', unit_scale=True):
            wrote = wrote + len(data)
            buffer.write(data)

        # If the file_path parameter is set, the content of the buffer will be saved too
        if not file_path is None:
            with open(file_path, 'wb+') as f:
                f.write(buffer.getvalue())

        # If the total size is not equal with the wrote size, the error is logged
        if total_size != 0 and wrote != total_size:
            logger.error("Something went wrong while downloading url: {}".format(url))
            return None
        return buffer

    def get_url(self, data_type: str, chunk_date: date, url_parser, district: int=None):
        """
        Generates the appropriate url for the input parameters. The url has to be downloaded from official PeMS
        site, thus this method caches all downloaded urls to minimize the url requests to PeMS site.

        :param url_parser:
        :param data_type:
        :param chunk_date:
        :param district:
        :return:
        """
        if district is None:
            district = "all"

        url_id = "{}_{}_{}".format(data_type, chunk_date, district)
        if url_id in self._chunk_urls:
            return str(self.SITE_URL+self._chunk_urls[url_id][1:])

        params = {
            "srq": "clearinghouse",
            "yy": chunk_date.year,
            "type": data_type,
            "returnformat": "text",
            "district_id": district
        }

        response_data: dict = self.session.get(url=self.SITE_URL, params=params).json()

        if isinstance(response_data, list) or not 'data' in response_data.keys():
            logger.warning("Missing data for {} in district {} at {}".format(data_type, district, chunk_date))
            return None

        self._chunk_urls.update(url_parser(response_data["data"], district))

        if not url_id in self._chunk_urls:
            logger.warning("Missing url for {} data in district {} at {}".format(data_type, district, chunk_date))
            return None

        return str(self.SITE_URL+self._chunk_urls[url_id][1:])

    def __str__(self):
        if self.initialized:
            return "PeMSConnection, logged in as {}".format(self._username)
        else:
            return "PeMSConnection, not initialized."

    def __repr__(self):
        return self.__str__()


class DataSourceHandler(metaclass=abc.ABCMeta):
    """
    The DataSourceHandler abstract class is responsible for handling the data loading
    """

    def __init__(self, **kwargs):

        if 'use_cache' in kwargs.keys():
            self.use_cache = kwargs['use_cache']
        else:
            self.use_cache = False

    @property
    def path(self):
        """
        Returns with the import directory of the DataSourceHandler.

        :return:
        """
        return os.path.join(os.getcwd(), get_imported_folder_path(), self.name)

    def load_between(self, from_date: date, to_date: date, district: int=None):
        """
        Loads all available data between the input dates from the selected district. In case of some data type, district
        information are not available, thus district parameter has to be leaved empty.

        :param from_date:
        :param to_date:
        :param district:
        :return:
        """
        os.makedirs(self.path, exist_ok=True)
        for chunk_date in self._chunk_dates(from_date, to_date):
             yield self._load_chunk_date(chunk_date, district=district)

    def _load_chunk_date(self, chunk_date: date, district: int):
        """
        Loads in one data chunk based on input parameters.

        :param chunk_date:
        :param district:
        :return:
        """
        file_path = self._file_path(chunk_date=chunk_date, district=district)

        # If the file already exists and the cache flag is not active the chunk will be loaded from the local storage
        if os.path.exists(file_path) and self.use_cache:
            logger.info("Loading imported data from: {}".format(file_path))
            return self._load_file(file_path)

        # Otherwise chunk will be loaded from the PeMSConnection.
        logger.info("Downloading {} data chunk from PeMS site at date {}".format(self.name, chunk_date))
        buffer = self._download_chunk(chunk_date=chunk_date, district=district, file_path=file_path if self.use_cache else None)
        if buffer is None:
            logger.error("Missing {} data chunk on PeMS site at date {}".format(self.name, chunk_date))
            return None
        logger.info("Loading downloaded {} data from memory".format(self.name))
        chunk = self._load_file(buffer)

        # It is important to close the buffer after the usage.
        buffer.close()
        if chunk is None:
            logger.error("Error during loading {} data chunk at date: {}".format(self.name, chunk_date))
        return chunk

    def _file_path(self, chunk_date: date, district: int) -> str:
        """
        Returns with the unique file path based on the input parameters.

        :param chunk_date:
        :param district:
        :return:
        """
        year = chunk_date.year
        month = chunk_date.month
        day = chunk_date.day
        district = "all" if district is None else str(district).zfill(3)
        day = "00" if day is None else str(day).zfill(2)
        return os.path.join(self.path, "{}_{}_{}_{}_d{}.txt.gz".format(self.name, year, month, day, district))

    def _download_chunk(self, chunk_date: date, file_path: str=None, district: int=None):
        """
        Downloads the data chunk from PeMS site. If the desired chunk does not exist in the database it returns with
        False, otherwise the return value is True.

        :param chunk_date: The date of the data chunk
        :param file_path:
        :param district:
        :return:
        """
        url = PeMSConnection().get_url(self.name, chunk_date, self._url_parser, district)
        return PeMSConnection().download(url, file_path)

    @abc.abstractmethod
    def _chunk_dates(self, from_datetime: date, to_datetime: date) -> Generator[date, Any, None]:
        """
        Returns whit list of dates have to be loaded. If the subjects of the investigation are months, the first day of
        the month have to be used.

        :param from_datetime:
        :param to_datetime:
        :return:
        """
        ...

    @abc.abstractmethod
    def _url_parser(self, url_list, district):
        """

        :param url_list:
        :return:
        """
        ...

    @abc.abstractmethod
    def _load_file(self, file_path) -> pd.DataFrame:
        """
        Loads a pd.DataFrame from input file_path. The input file_path refers to a data chunk.

        :param file_path:
        :return: The loaded pd.DataFrame which has the right column names.
        :rtype: pd.DataFrame
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns with the data type name.

        :return: Name of the data type.
        :rtype: str
        """
        ...

    def __str__(self):
        return "{} data handler (use_cache={})".format(self.name, self.use_cache)

    def __repr__(self):
        return self.__str__()


class StationRawDataHandler(DataSourceHandler):

    @property
    def name(self) -> str:
        return "station_raw"

    def _chunk_dates(self, from_datetime: date, to_datetime: date) -> Generator[date, Any, None]:
        return date_range(from_datetime, to_datetime)

    def _url_parser(self, url_list, district):
        processed_urls = {}
        for monthly_data in url_list.items():
            for recorded_day in monthly_data[1]:
                recorded_date = filename_to_date_1(recorded_day["file_name"])
                processed_urls["{}_{}_{}".format(self.name, recorded_date, district)] = recorded_day["url"]
        return processed_urls

    def _load_file(self, file_or_buffer):
        try:
            if isinstance(file_or_buffer, io.BytesIO):
                file_or_buffer.seek(0)
                zip_file = gzip.GzipFile(fileobj=file_or_buffer)
            else:
                zip_file = gzip.GzipFile(file_or_buffer)
            df = pd.read_csv(zip_file, header=None, index_col=False, parse_dates=[0], infer_datetime_format=True)
            new_column = ['timestamp', 'station_id']
            for i in range(0, int((len(df.columns) - 2) / 3)):
                new_column += [
                    'lane_{}_flow'.format(i),
                    'lane_{}_occupancy'.format(i),
                    'lane_{}_speed'.format(i),
                ]
            df.columns = new_column
            return df

        except zipfile.BadZipFile as err:
            logger.error("The data chunk cannot be unzipped. Error message: {}".format(err.args[0]))
            return None


class StationMetaDataHandler(DataSourceHandler):

    @property
    def name(self) -> str:
        return "metadata"

    def _chunk_dates(self, from_datetime: date, to_datetime: date) -> Generator[date, Any, None]:
        return date_range(from_datetime, to_datetime)

    def _url_parser(self, url_list, district):
        processed_urls = {}
        for data_record in url_list.items():
            record = data_record[1]
            dates = date_range(
                datetime.strptime(record['from_day'], "%m/%d/%y"),
                datetime.strptime(record['to_day'], "%m/%d/%y"))
            for a_date in dates:
                processed_urls["{}_{}_{}".format(self.name, a_date.date(), district)] = record["format"]["text"]["url"]
        return processed_urls

    def _load_file(self, file_or_buffer) -> pd.DataFrame:
        try:
            if isinstance(file_or_buffer, io.BytesIO):
                file_or_buffer.seek(0)

            df = pd.read_csv(file_or_buffer, sep="\t", compression=None)
            df = df[[df.columns[x] for x in range(0, 14)]]
            df.columns = ["id", "fwy", "dir", "district", "county", "city", "state_pm", "abs_pm", "latitude",
                          "longitude",
                          "length", "type", "lanes", "name"]
            return df

        except zipfile.BadZipFile:
            logger.error("The data chunk cannot be unzipped.")
            return None
        

class Station5MinDataHandler(DataSourceHandler):

    def _chunk_dates(self, from_datetime: date, to_datetime: date) -> Generator[date, Any, None]:
        return date_range(from_datetime, to_datetime)

    def _url_parser(self, url_list, district):
        processed_urls = {}
        for monthly_data in url_list.items():
            for recorded_day in monthly_data[1]:
                recorded_date = filename_to_date_1(recorded_day["file_name"])
                processed_urls["{}_{}_{}".format(self.name, recorded_date, district)] = recorded_day["url"]
        return processed_urls

    def _load_file(self, file_or_buffer) -> Optional[pd.DataFrame]:
        try:
            if isinstance(file_or_buffer, io.BytesIO):
                file_or_buffer.seek(0)
                zip_file = gzip.GzipFile(fileobj=file_or_buffer)
            else:
                zip_file = gzip.GzipFile(file_or_buffer)
            df = pd.read_csv(zip_file, header=None, index_col=False, parse_dates=[0], infer_datetime_format=True)
            new_column = ['timestamp', 'station_id', 'district', 'fwy_no', 'dir', 'lane_type', 'station_length',
                          'sample_no', 'obs_percentage', 'total_flow', 'avg_occupancy', 'avg_speed']
            for i in range(0, int((len(df.columns) - 12) / 5)):
                new_column += [
                    f'lane_{i}_samples',
                    f'lane_{i}_flow',
                    f'lane_{i}_avg_occ',
                    f'lane_{i}_avg_speed',
                    f'lane_{i}_observed'
                ]
            df.columns = new_column
            return df

        except zipfile.BadZipFile as err:
            logger.error("The data chunk cannot be unzipped. Error message: {}".format(err.args[0]))
            return None

    @property
    def name(self) -> str:
        return 'station_5min'
    
    
class StationHourDataHandler(DataSourceHandler):

    def _chunk_dates(self, from_datetime: date, to_datetime: date) -> Generator[date, Any, None]:
        return date_range(from_datetime, to_datetime)

    def _url_parser(self, url_list, district):
        processed_urls = {}
        for monthly_data in url_list.items():
            for recorded_day in monthly_data[1]:
                recorded_date = filename_to_date_2(recorded_day["file_name"])
                processed_urls["{}_{}_{}".format(self.name, recorded_date, district)] = recorded_day["url"]
        return processed_urls

    def _load_file(self, file_or_buffer) -> Optional[pd.DataFrame]:
        try:
            if isinstance(file_or_buffer, io.BytesIO):
                file_or_buffer.seek(0)
                zip_file = gzip.GzipFile(fileobj=file_or_buffer)
            else:
                zip_file = gzip.GzipFile(file_or_buffer)
            df = pd.read_csv(zip_file, header=None, index_col=False, parse_dates=[0], infer_datetime_format=True)
            new_column = ['timestamp', 'station_id', 'district', 'route_no', 'dir', 'lane_type', 'station_length',
                          'sample_no', 'obs_percentage', 'total_flow', 'avg_occupancy', 'avg_speed', 'delay_35', 
                          'delay_40', 'delay_45', 'delay_50', 'delay_55', 'delay_60']
            for i in range(0, int((len(df.columns) - 18) / 3)):
                new_column += [
                    f'lane_{i}_flow',
                    f'lane_{i}_avg_occ',
                    f'lane_{i}_avg_speed'
                ]
            df.columns = new_column
            return df

        except zipfile.BadZipFile as err:
            logger.error("The data chunk cannot be unzipped. Error message: {}".format(err.args[0]))
            return None

    @property
    def name(self) -> str:
        return 'station_hour'